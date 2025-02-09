import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from API import get_teacher_action, get_teacher_score
import requests
import json
import re
import PIL.Image
import math
import yaml
import re

def extract_coordinates(action):
    """Extracts coordinates from an action string in the format 'CLICK <point>[[x,y]]</point>'."""
    start = action.find("[[") + 2
    end = action.find("]]")
    coordinates = action[start:end].split(",")
    return float(coordinates[0]), float(coordinates[1])

class QwenAgent:
    def __init__(self, device, accelerator, cache_dir='~/.cache', dropout=0.5, policy_lm=None,
                 max_new_tokens=32, use_bfloat16=False):
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            policy_lm,  torch_dtype="auto", device_map="balanced"
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(policy_lm)
        self.tokenizer = AutoTokenizer.from_pretrained(policy_lm, trust_remote_code=True, cache_dir=cache_dir)
        self.tokenizer.truncation_side = 'left'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.device = device
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.accelerator = accelerator
        self.max_new_tokens = max_new_tokens
 

  
    def prepare(self): 
        self.model = self.accelerator.prepare(self.model)


    def get_osaction_and_score(self, obs):
        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs and give a score. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
       
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

         Basic Action 3: SCROLL
            - Purpose: SCROLL in the specified direction.
            - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - Example Usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: ENTER
            - purpose: Press the enter button.
            - format: ENTER
            - example usage: ENTER

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

    In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.
        
	And I hope you evaluate your action to be scored, giving it a score from 1 to 5. 
    A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.
    1 means you believe this action definitely cannot achieve the goal.
    2 means you believe this action is very unlikely to achieve the goal.
    3 means you believe this action has a certain chance of achieving the goal.
    4 means you believe this action is very likely to achieve the goal.
    5 means you believe this action will definitely achieve the goal.

	And your final goal, previous actions and associated screenshot are as follows:
    Final goal: {obs['task']}
    previous actions: {obs['previous_actions']}
    Screenshot: 
        
	Your output must strictly follow the format below , and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas_action must be one of the action formats I provided and score must be 1 to 5):
    action: 
    score:
    """
                 
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": sys_prompt,
                            },
                            {
                                "type": "image",
                                "image": obs['image_path'],
                            },
                        ],
                    }
                ]

        
        chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        self.device = self.model.device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128).to(self.device)
        generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

        print(output_text[0])
        action_pattern = r"action: (.+)"
        score_pattern = r"score:(\d+)"

  
        action = re.search(action_pattern, output_text[0]).group(1)
        score = re.search(score_pattern, output_text[0]).group(1)

        score = int(score)
        #print("Action:", action)
        #print("Score:", score)

        return action, score

    def _get_a_action(self, obs):
        sys_prompt = f"""
        You are now operating in Executable Language Grounding mode. Your goal is to help users accomplish tasks by suggesting executable actions that best fit their needs. Your skill set includes both basic and custom actions:

        1. Basic Actions
        Basic actions are standardized and available across all platforms. They provide essential functionality and are defined with a specific format, ensuring consistency and reliability. 
        Basic Action 1: CLICK 
            - purpose: Click at the specified position.
            - format: CLICK <point>[[x-axis, y-axis]]</point>
            - example usage: CLICK <point>[[101, 872]]</point>
       
        Basic Action 2: TYPE
            - purpose: Enter specified text at the designated location.
            - format: TYPE [input text]
            - example usage: TYPE [Shanghai shopping mall]

         Basic Action 3: SCROLL
            - Purpose: SCROLL in the specified direction.
            - Format: SCROLL [direction (UP/DOWN/LEFT/RIGHT)]
            - Example Usage: SCROLL [UP]
            
        2. Custom Actions
        Custom actions are unique to each user's platform and environment. They allow for flexibility and adaptability, enabling the model to support new and unseen actions defined by users. These actions extend the functionality of the basic set, making the model more versatile and capable of handling specific tasks.

        Custom Action 1: PRESS_BACK
            - purpose: Press a back button to navigate to the previous screen.
            - format: PRESS_BACK
            - example usage: PRESS_BACK

        Custom Action 2: PRESS_HOME
            - purpose: Press a home button to navigate to the home page.
            - format: PRESS_HOME
            - example usage: PRESS_HOME

        Custom Action 3: COMPLETE
            - purpose: Indicate the task is finished.
            - format: COMPLETE
            - example usage: COMPLETE

        Custom Action 4: IMPOSSIBLE
            - purpose: Indicate the task is impossible.
            - format: IMPOSSIBLE
            - example usage: IMPOSSIBLE

        In most cases, task instructions are high-level and abstract. Carefully read the instruction and action history, then perform reasoning to determine the most appropriate next action.
        And your previous actions, current task instruction, step list and associated screenshot are as follows:
        Final goal: {obs['task']}
        previous actions: {obs['previous_actions']}
        Screenshot:
        Your output must be in one line. Do not split it into two lines. 
        Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas action must be one of the action formats I provided):
        action:

    """  
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": sys_prompt,
                            },
                            {
                                "type": "image",
                                "image": obs['image_path'],
                            },
                        ],
                    }
                ]

        # 处理输入并生成
        chat_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
                    text=[chat_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        self.device = self.model.device
        inputs = inputs.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128).to(self.device)
        generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )


        prefix = 'actions:\n'
        start_index = output_text[0].find(prefix) + len(prefix)
        result = output_text[0][start_index:]

        #print(output_text[0])
        #action_pattern = r"action: (.+)"
        #result = re.search(action_pattern, output_text[0]).group(1)

        print(result)
        return result


    def get_action(self, obs):
        result = {}

        with open('./config/config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if config['mode'] == 'makedata_mode':
            teacher_action = get_teacher_action(obs['task'], obs['list'][obs['now_step']], obs['image_path'], obs['ocr'], obs['list'], obs['previous_actions'])
            osatlas_action = self._get_a_action(obs)
            score = get_teacher_score(obs['task'], obs['list'][obs['now_step']], obs['image_path'], osatlas_action, obs['ocr'], teacher_action, obs['previous_actions'])
        
            if teacher_action.startswith("CLICK") and osatlas_action.startswith("CLICK"):
                x1, y1 = extract_coordinates(teacher_action)
                x2, y2 = extract_coordinates(osatlas_action)

                # Calculate Euclidean distance
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if distance > 140:
                    score = score - 2
                    if score < 1 :
                        score = 1
                elif distance > 70:
                    score = score - 1
                    if score < 1 :
                        score = 1
                elif distance < 70:
                    score = 5
                elif distance > 70 & distance < 140:
                    score = 4
            if osatlas_action.startswith("PRESS_HOME"):
                score = score - 2
                if score < 1 :
                    score = 1
            if osatlas_action.startswith("COMPLETE"):
                score = score - 2
                if score < 1 :
                    score = 1
            if teacher_action.startswith("IMPOSSIBLE"):
                if not osatlas_action.startswith("IMPOSSIBLE"):
                    score = score - 4
                    if score < 1 :
                        score = 1
            if osatlas_action.startswith("SCROLL"):
                if teacher_action.startswith("CLICK"):
                    score = score - 2
                    if score < 1 :
                        score = 1
            if osatlas_action.startswith("TYPE"):
                if teacher_action.startswith("TYPE"):
                    if osatlas_action!=teacher_action:
                        score = score - 2
                        if score < 1 :
                            score = 1   
            if  teacher_action.startswith("TYPE"):   
                if not osatlas_action.startswith("TYPE"):
                    score = score - 2
                    if score < 1 :
                        score = 1      
                                           
            print(score)        
            result['teacher_action'] = teacher_action
            result['osatlas_action'] = osatlas_action
            result['score'] = score
        
        elif config['mode'] == 'test_mode' and (config['test_mode'] == 'realworld_test' or config['test_mode'] == 'gpt_test'):
            result['osatlas_action'], result['score'] = self.get_osaction_and_score(obs)
            if (config['test_mode'] == 'single_test' or config['test_mode'] == 'gpt_test'):
                teacher_action = get_teacher_action(obs['task'], obs['list'][obs['now_step']], obs['image_path'], obs['ocr'], obs['list'], obs['previous_actions'])
                result['teacher_action'] = teacher_action
        
        elif config['mode'] == 'test_mode' and config['test_mode'] == 'the_entire_trajectory':
            osatlas_action = self._get_a_action(obs)
            result['osatlas_action'] = osatlas_action
        
        return result
