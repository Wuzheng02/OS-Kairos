import numpy as np
import os
import torch
import time
import json
import math
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import PIL.Image
import math
import re
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from ocr_util import get_ocr
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI
import base64
class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7


def qwen_translate_action(out):
    if out == "PRESS_BACK":
        return AndroidAction(action_type=ActionType.GoBack)
    elif out == "PRESS_HOME":
        return AndroidAction(action_type=ActionType.GoHome)
    elif out == "ENTER":
        return AndroidAction(action_type=ActionType.Enter)
    elif out == "COMPLETE":
        return AndroidAction(action_type=ActionType.TaskComplete)
    elif out == "IMPOSSIBLE":
        return AndroidAction(action_type=ActionType.TaskImpossible)
    elif out == "SCROLL [RIGHT]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.2, 0.5), lift_point=(0.8, 0.5))
    elif out == "SCROLL [LEFT]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.8, 0.5), lift_point=(0.2, 0.5))
    elif out == "SCROLL [UP]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.5), lift_point=(0.5, 0.2))
    elif out == "SCROLL [DOWN]":
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=(0.5, 0.2), lift_point=(0.5, 0.5))
    elif out.startswith("TYPE [") and out.endswith("]"):
        start = out.find("[") + 1
        end = out.find("]")
        text = out[start:end]
        return AndroidAction(action_type=ActionType.Type, typed_text=text)
    elif out.startswith("CLICK <point>[[") and out.endswith("]]</point>"):
        point_str = out.split("<point>[[")[1].split("]]</point>")[0]
        point_values = point_str.split(",")        
        x_axis = float(point_values[0].strip()) /1000
        y_axis = float(point_values[1].strip()) /1000
        touch_point=(x_axis, y_axis)
        return AndroidAction(action_type=ActionType.DualPoint, touch_point=touch_point, lift_point=touch_point)


def Is_action_success(raw_action1, raw_action2):
    action1 = qwen_translate_action(raw_action1)
    action2 = qwen_translate_action(raw_action2)
    if action1.action_type != action2.action_type:
        return False
    elif action1.action_type == ActionType.Type:
        if action1.typed_text != action2.typed_text:
            return False
    elif action1.action_type == ActionType.DualPoint:
        if raw_action1.startswith("SCROLL"):
            if not raw_action2.startswith("SCROLL"):
                return False                
        if raw_action2.startswith("SCROLL"):
            if not raw_action1.startswith("SCROLL"):
                return False 
        if raw_action1.startswith("SCROLL"):
            if raw_action2.startswith("SCROLL"):
                if raw_action1!=raw_action2:
                    return False 
        else:
            def extract_coordinates(action):
                """Extracts coordinates from an action string in the format 'CLICK <point>[[x,y]]</point>'."""
                start = action.find("[[") + 2
                end = action.find("]]")
                coordinates = action[start:end].split(",")
                return float(coordinates[0]), float(coordinates[1])
            x1, y1 = extract_coordinates(raw_action1)
            x2, y2 = extract_coordinates(raw_action2)
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if distance > 140:
                return False
    return True

def Is_action_type(raw_action1, raw_action2):
    action1 = qwen_translate_action(raw_action1)
    action2 = qwen_translate_action(raw_action2)
    #print(action1)
    #print(action2)
    if action1.action_type != action2.action_type:
        return False
    else :
        return True

def test_loop(test_path, ocr_detection, ocr_recognition):
    with open(test_path, 'r') as file:
        data = json.load(file)

    element_count = 0
    success_count = 0
    type_true_count = 0

    episode_conut = 0
    success_episode_count = 0

    now_task = data[0]['task']
    
    signal = True
    episode_conut = episode_conut + 1


    for obs in data:
        try:
            if now_task != obs['task']:
                now_task = obs['task']
                episode_conut = episode_conut + 1
                if signal == True:
                    success_episode_count = success_episode_count + 1
                signal = True

            obs['ocr'] = get_ocr(obs['image_path'], ocr_detection, ocr_recognition)

            action = get_gpt_action(obs['task'], obs['list'][obs['now_step']], obs['image_path'], obs['ocr'], obs['list'], obs['previous_actions'])
            element_count = element_count + 1
            print(action)

            if Is_action_success(action, obs['teacher_action']):
                success_count = success_count + 1
            else:
                signal = False

            if Is_action_type(action, obs['teacher_action']):
                type_true_count = type_true_count + 1
            print(element_count)
            print(success_count)
            print(type_true_count)



        except Exception as e:
            print(f"Error processing observation {element_count}: {e}")
            continue  

    step_rate = success_count / element_count
    type_rate = type_true_count / element_count
    episode_success_rate = success_episode_count / episode_conut
    print(step_rate)
    print(type_rate)
    print(episode_success_rate)



def extract_coordinates(action):
    """Extracts coordinates from an action string in the format 'CLICK <point>[[x,y]]</point>'."""
    start = action.find("[[") + 2
    end = action.find("]]")
    coordinates = action[start:end].split(",")
    return float(coordinates[0]), float(coordinates[1])

def get_gpt_action(final_goal, current_goal, image_path, ocr, step_list, previous_actions):
    os.environ['http_proxy'] = 'http://127.0.0.1:7900'
    os.environ['https_proxy'] = 'http://127.0.0.1:7900'
    prompt = (
    "### Background ###\n"
    "You are an expert in completing tasks based on screenshots and instructions. "
    "I will provide you with a mobile screenshot, a final goal, the current goal, the previous actions and the step list. "
    "Based on the mobile screenshot, the final goal, the current goal and the step list. I need you to determine the action to take. "
    "The Current Goal may not be accurate, but the correct Current Goal must be one of the steps in the step list. If you feel that the Current Goal is not accurate, please use the step list to determine the appropriate Current Goal to execute."
    f"Final Goal: {final_goal}\n"
    f"Current Goal: {current_goal}\n"
    f"previous actions : {previous_actions}"
    f"step list: {step_list}\n"
    "### Screenshot information ###\n"
    "To help you understand the information in the screenshot, I first performed OCR. Here are the names and coordinates of the icons obtained through OCR:"
    f"Coordinates of the icons: {ocr}"
    "### Response requirements ###\n"
    "Your skill set includes both basic and custom actions:\n"
    "1. Basic Actions\n"
    "Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability.\n"
    "Basic Action 1: CLICK\n"
    "    - Purpose: Click at the specified position.\n"
    "    - Format: CLICK <point>[[x-axis,y-axis]]</point>\n"
    "    - Example Usage: CLICK <point>[[101,872]]</point>\n"
    "    - Tips:The x-coordinate represents the thousandth part of the screen's width, counted from left to right.The y-coordinate represents the thousandth part of the screen's height, counted from top to bottom.Obviously, the range of both x and y is [0, 1000].\n\n"
    "Basic Action 2: TYPE\n"
    "    - Purpose: Enter specified text at the designated location.\n"
    "    - Format: TYPE [input text]\n"
    "    - Example Usage: TYPE [Shanghai shopping mall]\n\n"
    "Basic Action 3: SCROLL\n"
    "    - Purpose: SCROLL in the specified direction.\n"
    "    - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]\n"
    "    - Example Usage: SCROLL [UP]\n\n"
    "2. Custom Actions\n"
    "Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.\n"
    "Custom Action 1: PRESS_BACK\n"
    "    - Purpose: Press a back button to navigate to the previous screen.\n"
    "    - Format: PRESS_BACK\n"
    "    - Example Usage: PRESS_BACK\n\n"
    "Custom Action 2: PRESS_HOME\n"
    "    - Purpose: Press a home button to navigate to the home page.\n"
    "    - Format: PRESS_HOME\n"
    "    - Example Usage: PRESS_HOME\n\n"
    "Custom Action 3: COMPLETE\n"
    "    - Purpose: Indicate the task is finished.\n"
    "    - Format: COMPLETE\n"
    "    - Example Usage: COMPLETE\n\n"
    "Custom Action 4: IMPOSSIBLE\n"
    "    - Purpose: Indicate the task is impossible.\n"
    "    - Format: IMPOSSIBLE\n"
    "    - Example Usage: IMPOSSIBLE\n\n"
    "### Output format ###\n"
    "Your response must exactly follow the template:\n"
    "{action: ACTION_NAME}\n"
    "Replace `ACTION_NAME` with one of:\n"
    "- CLICK <point>[[x,y]]</point>\n"
    "- TYPE [input text]\n"
    "- SCROLL [UP/DOWN/LEFT/RIGHT]\n"
    "- PRESS_BACK\n"
    "- PRESS_HOME\n"
    "- ENTER\n"
    "- IMPOSSIBLE\n"
    )
    

    client = OpenAI(
        base_url="https://api.gpts.vin/v1",
        api_key= "sk-nO4o2zE3reOUJRVq4BS4xPW0CqoniTIwl7Cys11i2ce1zzs"
    )
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                                },
                        },
                    ],
                }
            ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    chat_response = completion
    answer = chat_response.choices[0].message.content

    start = answer.find('action:') + 8
    end = answer.rfind('}')
    action = answer[start:end]
    print(action)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    return action

test_path = "/data1/wuzh/pre_experiment/test.json"
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

test_loop(test_path, ocr_detection, ocr_recognition)
