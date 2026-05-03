import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "your_visible_cuda_id"
sys.path.append('/your_local_path/CharLuMA')

import argparse
import torch
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

model_path = "/your_local_path/CharLuMA-1.3B"
image_file = "/your_local_path/test_inference.jpg"
question = "<image>\nYou are a skilled developer specializing in writing plotting code based on a given picture. I found a very nice picture in a paper, but there is no corresponding source code available. I need your help to generate the Python code that can reproduce the picture based on the picture I provide.\nNow, please give me the plotting code that reproduces the picture below, starting with \"```python\\n\" and ending with \"\\n```\"."
lang_type = 'python'

# Load the model
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Prepare for inference
prompt = f"### Instruction:\n{question}\n### Response:\n"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
image = Image.open(image_file).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)[0]
images = image_tensor.unsqueeze(0).half().cuda()
image_sizes = [image.size]

# Run inference
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=images,
        image_sizes=image_sizes,
        lang_type=[lang_type],
        do_sample=False,
        max_new_tokens=2048,
        use_cache=True,
    )
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(outputs)