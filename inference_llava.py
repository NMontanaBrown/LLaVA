# coding=utf-8

"""

"""

import tqdm
import pandas as pd
import random
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import os
import json
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import torch

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from llava.conversation import conv_templates, SeparatorStyle

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def inference_one_qa(tokenizer,
                     model,
                     image_processor,
                     datum,
                     path_images,
                     temperature=0.2,
                     max_new_tokens=512):
    """
    For a given item in the test dataset json,
    generate a response using the model.

    """

    conv_mode = "llava_v1"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    messages = datum["conversations"]

    results = {}
    results["id"] = datum["image"]

    human_messages = [m for m in messages if m["from"] == "human"][0]["value"].replace("<image>\n", "")
    ground_truth = [m for m in messages if m["from"] == "gpt"][0]["value"]

    results["question"] = human_messages
    results["ground_truth"] = ground_truth

    inp = f"{roles[0]}: "

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    inp = inp + human_messages
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    image = load_image(os.path.join(path_images, datum["image"]))
    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)

    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True)

    outputs = tokenizer.decode(output_ids[0]).strip()
    results["output"] = outputs
    
    return results


if __name__ == "__main__":
    path_test = "/home/ubuntu/cxrdata/conversation_dataset_test_full.json"
    path_images = "/home/ubuntu/cxrdata/flat_images/flat_images/"
    model_path = "/home/ubuntu/repos/LLaVA/checkpoints/llava-v1.5-7b-ft-full-rg"
    model_base = "lmsys/vicuna-7b-v1.5"

    model_name = get_model_name_from_path(model_path)
    temperature = 0.0
    max_new_tokens = 512

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

    with open(path_test, "r") as f:
        data = json.loads(f.read())

    data_images_unique = list(set([d["image"] for d in data]))
    # randomly sample 500
    random.seed(42)
    data_images_unique = random.sample(data_images_unique, 500)

    # get subset of data that has the same image ids
    pd_data = pd.DataFrame(data)
    print(pd_data.head())
    print(pd_data.columns)
    pd_data = pd_data[pd_data["image"].isin(data_images_unique)]
    # convert back to list of dicts
    data = list(pd_data.to_dict(orient="records"))

    results = []
    for datum in tqdm.tqdm(data):
        results.append(inference_one_qa(tokenizer,
                                         model,
                                         image_processor,
                                         datum,
                                         path_images,
                                         temperature=temperature,
                                         max_new_tokens=max_new_tokens))
    
    pd_results = pd.DataFrame(results)
    pd_results.to_csv("results_patch_llava_constr.csv", index=False)