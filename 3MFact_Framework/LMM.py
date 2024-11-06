

import os
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import random

import requests


import time
import logging
import json
import re
import regex
import cv2

model_path = '/Data/niukaipeng/pretrained_model/OpenBMB/MiniCPM-V-2_6'

# Load model and tokenizer once
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, 
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16) 
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number

def uniform_sample(l, n):
    gap = len(l) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [l[i] for i in idxs]

def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print(f'Video: {video_path}, num frames: {len(frames)}')
    return frames

def process_image(image_path, question):
    image = Image.open(image_path).convert('RGB')
    msgs = [{'role': 'user', 'content': [image, question]}]

    # Process the image
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    return answer

def analysis_video_minicpm(video_path, question):
    frames = encode_video(video_path)
    msgs = [{'role': 'user', 'content': frames + [question]}]
    
    # Set decode params for video
    params = {
        "use_image_id": False,
        "max_slice_nums": 1  # use 1 if cuda OOM and video resolution > 448*448
    }

    # Process the video
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        **params
    )
    
    return answer



