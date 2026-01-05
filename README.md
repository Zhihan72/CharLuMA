# Aligned Multi-View Scripts for Universal Chart-to-Code Generation

This repository is provided for the **double-blind reviewing process of ARR 2026 January Cycle**.

**Notice:**
* **Anonymity:** The authors' identities and affiliations are strictly hidden.
* **License:** Redistribution of any data or code contained here is strictly prohibited during the review period.

---

## Introduction

Chart-to-code generation converts a chart image into an executable plotting script, enabling faithful reproduction and editable visualizations. Existing methods are largely Python-centric, limiting practical use and overlooking a key source of supervision: the same chart can be expressed by semantically equivalent scripts in different plotting languages.

To address this, we introduce:

1.  **Chart2NCode:** A dataset of 176K charts paired with aligned scripts in Python, R, and LaTeX. These scripts render visually equivalent outputs and were constructed via a metadata-to-template pipeline with rendering verification.
2.  **CharLuMA:** A parameter-efficient adaptation module built on a LLaVA-style MLLM. It augments the multimodal projector with a language-conditioned mixture of low-rank subspaces, allowing the model to share core chart understanding while specializing code generation to the target language through lightweight routing.

We release the following resources:
* Automatic annotation pipeline (under `dataset_construction/`) and a subset of Chart2NCode (under `dataset_construction/sample_Chart2NCode/`).
* Model architecture codes (under `llava/`) and training scripts (under `scripts/`).

## Dataset Construction

To replicate the automatic annotation pipeline used to generate Chart2NCode, run:

```
bash dataset_construction/main.sh
```
Note: This release currently includes the templates and template-filling scripts for area, bar, and box charts.

We provide a random subset of the Chart2NCode dataset in `dataset_construction/sample_Chart2NCode/`. Due to storage constraints and the strict prohibition on external links during the review process, the full dataset cannot be hosted in this repository.

## Training

The training strategy consists of two stages: alignment pretraining and instruction tuning.

For **alignment pretraining**, run

```
bash scripts/pretrain_modality_alignment.sh
```

Instruction tuning is divided into a warm-up phase followed by the full training of the adapter and language model backbone. First, warm-up the model:

```
bash scripts/finetune_warmup.sh
```

Then, train the adapter and language model backbone:

```
bash scripts/finetune_instruction_tuning.sh
```

The **core implementation of the CharLuMA architecture**, including the novel parameter-efficient adaptation module (language-conditioned mixture of low-rank subspaces), can be found in the following file: `llava/model/multimodal_projector/mlp_murmoe.py`.

## Acknowledgement

The model architecture and training scripts are built upon LLaVA. We thank the authors for their contributions to the open-source community.
