

import logging
# 首先清空日志文件
open('main_minicpm.log', 'w').close()
# 然后配置日志系统
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='main_minicpm.log', filemode='a')



import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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

model_path = 'OpenBMB/MiniCPM-V-2_6'

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

def process_video(video_path, question):
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











def gpt35_summary(video_llava_answer, key_frame_highlights):
    # api_key = ''
    api_key = ''
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    params = {
    "messages": [
        {
            "role": "system",
            "content": """
            Your task is to generate a coherent, logically structured, and accurate video description. This description must:
            1. Strictly adhere to the provided information, with absolutely no speculation or additions.
            2. Integrate the overall video content analysis with detailed information from 7 key frames.
            3. Maintain the highest level of accuracy as the paramount principle.
            4. Create a fluid, logically clear narrative that encompasses all critical details.
            5. Range between 100-500 words, ensuring comprehensiveness while avoiding redundancy.
            """
        },
        {
            "role": "user",
            "content": f"""
            Based on the following information, craft a cohesive and accurate video description:

            Overall video content: {video_llava_answer}

            Highlights from 7 key frames: {key_frame_highlights}

            Your description must:
            1. Synthesize the overall content and information from 7 key frames into a single, cohesive narrative.
            2. Adhere strictly to facts, with absolutely no speculation.
            3. Organize content chronologically or logically, ensuring narrative continuity and fluency.
            4. Include all significant actions, scenes, and visual element details.
            5. Maintain an objective and accurate tone throughout.
            6. Ensure each detail is directly supported by the provided information.
            7. Create a description comprehensible to someone who has not viewed the video.

            Final output: A logically clear, cohesive, and accurate video description encompassing the entire video content.

            Remember: Accuracy is the highest priority, followed by comprehensiveness and coherence.
            """
        }
    ],
    "model": "gpt-4o-mini"
}

    response = requests.post(
        "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False
    )
    res = response.json()

    video_summary_answer = res['choices'][0]['message']['content']


    return video_summary_answer



















def pipe_prompt_2_only_accuracy(video_file_path,image_folder_path):

    video_prompt = "Describe the key events in the video chronologically, including time, location, and participants. Focus on observable actions and processes, avoiding speculation. Provide a brief and accurate summary of the video content."

    image_prompt = "Accurately describe this image, including visible subjects, their actions, and the main scene or background information. Extract and describe any visible text. Only describe what can be directly observed in the image. Do not speculate on uncertain details, only describe the most certain elements."

    temperature = 0.2
    all_answers = {
    "Video": {},  # 初始化为字典
    "Image": {},       # 初始化为字典
    "GPT3.5":{}      # 初始化为字典
    }

    video_summary_answer = process_video(video_file_path, video_prompt)
    all_answers["Video"]["question"] = video_prompt
    all_answers["Video"]["answer"] = video_summary_answer

    image_summary_answer = {}
    
    all_answers["Image"]["question"] = image_prompt

    image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sorted(image_files)

    for image_path in sorted_image_files:
        image_name = os.path.join(image_folder_path, image_path)
        answer = process_image(image_name, image_prompt)
        all_answers["Image"][image_path] = {"answer": answer}
    

    key_frame_summary = "; ".join([
        f"{image_file}: {details['answer']}"
        for image_file, details in all_answers["Image"].items()
        if 'answer' in details and details['answer']  # 确保只包括有答案的条目
    ])

    gpt35_summary_answer = gpt35_summary(video_summary_answer, key_frame_summary)
    all_answers["GPT3.5"]["answer"] = gpt35_summary_answer

    return all_answers





















def __main__():


    dataset_folder = "/home/public/FakeNews/code/LLMFND/select_all_videos"
    # dataset_folder = "/home/public/FakeNews/code/LLMFND/test"
    result_folder = "/home/public/FakeNews/code/LLMFND/all_videos_description_minicpm"

    # 如果想要按字典序倒序排序，可以使用sorted的reverse参数
    video_names = sorted([file for file in os.listdir(dataset_folder) if file.endswith(".mp4") or file.endswith(".mkv")])
    

    
    
    i= 0
    for video_name in video_names:

        start_time = time.time()
        logging.info(f"Processing video: {video_name}")
        
        video_path = os.path.join(dataset_folder, video_name)


        video_dir, video_name = os.path.split(video_path)
        video_base_name = os.path.splitext(video_name)[0]
        image_folder_path = os.path.join(video_dir, video_base_name)

        # 使用正则表达式替换文件名中的 .mp4 或 .mkv 扩展名
        base_filename = re.sub(r'\.mp4$|\.mkv$', '', video_name)

        # 使用格式化的字符串拼接完整的文件路径
        json_file_path = os.path.join(result_folder, f"{base_filename}.json")

        # 先检查json文件是否已经存在，如果存在则跳过
        if os.path.exists(json_file_path):
            logging.info(f"File {json_file_path} already exists, skipping...")
            i+=1
            logging.info(f"i={i}")
            continue


        answer2 = pipe_prompt_2_only_accuracy(video_path, image_folder_path)



        final_answer = {
            "Zero-Shot Detailed Inquiry Prompt": answer2
        }

        # 将用户的答案保存到json文件中
        with open(json_file_path, 'w', encoding="utf-8-sig") as f:
            json.dump(final_answer, f)
        
        end_time = time.time()

        logging.info(f"Time taken for video {video_name}: {(end_time - start_time)/60} minutes.")

        logging.info(f"----------------------------- Analysis for {json_file_path} completed successfully. -----------------------------")


if __name__ == "__main__":
    try:
        __main__()
    except Exception as e:
        logging.exception("Exception occurred")
        logging.error(e)
        raise e