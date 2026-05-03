import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

class MLPCharLuMA(nn.Module):
    """
    mm_channels : int   Vision feature dim (e.g., 1152)
    channels    : int   LM hidden dim    (e.g., 4096)
    rank        : int   r selected subspaces per sample (<= N)
    N           : int   Total candidate B columns (N > r)
    alpha       : float LoRA scale for delta
    temp        : float Global router temperature
    freeze_A    : bool  Freeze A
    freeze_mlp  : bool  Freeze base MLP
    lang2id     : dict  Mapping, e.g., {"python":0,"latex":1,"r":2}
    languages   : list  Ordered list of language keys (len=3 by default)
    """
    def __init__(
        self,
        mm_channels: int,
        channels: int,
        rank: int = 16,
        N: int = 32,
        alpha: float = 16.0,
        temp: float = 1.0,
        freeze_A: bool = True,
        freeze_mlp: bool = True,
        lang2id: Optional[Dict[str, int]] = None,
        languages: Optional[List[str]] = None,
    ):
        super().__init__()
        assert N > rank, "N must be > rank."
        self.mm_channels = mm_channels
        self.channels = channels
        self.N = N
        self.r = rank
        self.alpha = alpha
        self.temp = temp

        if languages is None:
            languages = ["python", "latex", "r"]
        self.languages = languages
        self.num_langs = len(self.languages)
        self.lang2id = lang2id or {k: i for i, k in enumerate(self.languages)}

        self.mlp_connector = nn.Sequential(
            nn.Linear(mm_channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        if freeze_mlp:
            for p in self.mlp_connector.parameters():
                p.requires_grad = False
        
        self.A = nn.Linear(mm_channels, rank, bias=False)
        if freeze_A:
            for p in self.A.parameters():
                p.requires_grad = False
        
        # if freeze_A: # for initialization
        #     nn.init.orthogonal_(self.A.weight)

        self.B = nn.Parameter(torch.empty(channels, N))
        nn.init.zeros_(self.B)

        self.lang_Wq = nn.ModuleDict({
            lang: nn.Linear(mm_channels, N, bias=False) for lang in self.languages
        })
    
    def _lang_ids(self, lang_type, B: int, device) -> torch.LongTensor:
        if lang_type is None:
            return torch.zeros(B, dtype=torch.long, device=device)
        if isinstance(lang_type, str):
            return torch.full((B,), self.lang2id.get(lang_type.lower(), 0),
                              dtype=torch.long, device=device)
        ids = [self.lang2id.get(x.lower(), 0) for x in lang_type]
        if len(ids) == 1 and B > 1:
            ids = ids * B
        if len(ids) != B:
            raise ValueError("lang_type length must be 1 or equal to batch size.")
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _router_logits(self, h_bar: torch.Tensor, lang_ids: torch.LongTensor) -> torch.Tensor:
        Wq_stack = torch.stack(
            [self.lang_Wq[lang].weight.t() for lang in self.languages], dim=0
        ).to(h_bar.dtype).to(h_bar.device)
        Wq_sel = Wq_stack.index_select(0, lang_ids)
        logits = torch.bmm(h_bar.unsqueeze(1), Wq_sel).squeeze(1)
        return logits / self.temp
    
    def forward(self, image_features: torch.Tensor, lang_type: Optional[List[str]] = None) -> torch.Tensor:
        h = image_features
        Bsz, L, _ = h.shape

        base = self.mlp_connector(h)
        z = self.A(h)
        h_bar = h.mean(dim=1)
        lang_ids = self._lang_ids(lang_type, Bsz, h.device)

        logits = self._router_logits(h_bar, lang_ids)
        topk = torch.topk(logits, k=self.r, dim=-1)
        idx  = topk.indices
        vals = topk.values
        w    = F.softmax(vals, dim=-1)

        B_sel = self.B.t().index_select(0, idx.view(-1)) \
                          .view(Bsz, self.r, self.channels) \
                          .transpose(1, 2)
        wz = z * w.unsqueeze(1)
        delta = torch.bmm(wz, B_sel.transpose(1, 2))

        return base + delta * (self.alpha / self.r)
    
    @torch.no_grad()
    def peek(self, image_features: torch.Tensor, lang_type: Optional[List[str]] = None):
        Bsz, L, _ = image_features.shape
        h_bar = image_features.mean(dim=1)
        lang_ids = self._lang_ids(lang_type, Bsz, image_features.device)
        logits = self._router_logits(h_bar, lang_ids)
        topk = torch.topk(logits, k=self.r, dim=-1)
        idx = topk.indices
        w   = F.softmax(topk.values, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return {"idx": idx, "weights": w, "logits": logits, "probs": probs, "lang_ids": lang_ids}