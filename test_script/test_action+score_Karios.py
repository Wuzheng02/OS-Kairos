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
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

class ActionType(Enum):
    Idle=0
    DualPoint=1
    Type=2
    GoBack=3
    GoHome=4
    Enter=5
    TaskComplete=6
    TaskImpossible=7

@dataclass
class AndroidAction():
    action_type: ActionType
    touch_point: Tuple[float, float] = None
    lift_point: Tuple[float, float] = None
    typed_text: str = None

    def __str__(self):
        # Construct the basic action type string.
        components = [f"Action Type: {self.action_type.name}"]

        # Format and add touch_point if it's not None.
        if self.touch_point:
            touch_point_str = f"({self.touch_point[0]:.4f}, {self.touch_point[1]:.4f})"
            components.append(f"Touch Point: {touch_point_str}")
        if self.lift_point:
            lift_point_str = f"({self.lift_point[0]:.4f}, {self.lift_point[1]:.4f})"
            components.append(f"Lift Point: {lift_point_str}")
        if self.typed_text:
            components.append(f"Typed Text: '{self.typed_text}'")
        return ", ".join(components)

    def to_act(self):
        pass

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
        elif raw_action2.startswith("SCROLL"):
            if not raw_action1.startswith("SCROLL"):
                return False 
        elif raw_action1.startswith("SCROLL"):
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
        if (raw_action1.startswith("CLICK") and raw_action2.startswith("SCROLL")) or (raw_action2.startswith("CLICK") and raw_action1.startswith("SCROLL")):
            return False
        
        return True

def get_action_type(action):
    if action.startswith("CLICK"):
        return 1
    elif action.startswith("SCROLL"):
        return 2
    elif action.startswith("TYPE"):
        return 3
    elif action.startswith("PRESS_BACK"):
        return 4
    elif action.startswith("PRESS_HOME"):
        return 4
    elif action.startswith("ENTER"):
        return 4
    elif action.startswith("COMPLETE"):
        return 5
    elif action.startswith("IMPOSSIBLE"):
        return 5
    else: 
        return 0
    
def test_loop(agent, test_path):
    with open(test_path, 'r') as file:
        data = json.load(file)
    #CLICK → 1
    #SCROOL → 2
    #TYPE → 3
    #PRESS_BACK、PRESS_HOME、ENTER → PRESS → 4
    #COMPLETE、 IMPOSSIBLE → STOP → 5
    CLICK_count = 0
    SCROLL_count = 0
    TYPE_count = 0
    PRESS_count = 0
    STOP_count = 0
    CLICK_success = 0
    SCROLL_success = 0
    TYPE_success = 0
    PRESS_success = 0
    STOP_success = 0    
    CLICK_type = 0
    TYPE_type = 0

    interaction_count = 0

    total_type = 0
    
    episode_conut = 0
    success_episode_count = 0
    now_task = data[0]['task']
    now_success = data[0]['success']
    signal = True

    GP_count = 0
    GP_sum = 0
    GP_single = []

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    now_count = 0
    result = []
    for obs in data:
        try:
            GP_sum = GP_sum + 1
            if signal == True:
                GP_count = GP_count + 1
            now_result = {}
            now_result['id'] = now_count
            if now_task != obs['task'] or obs == data[-1]:
                if now_success == True or (obs == data[-1] and obs['success']==True) :
                    episode_conut = episode_conut + 1
                    GP_single.append(GP_count / GP_sum)
                    GP_count = 0
                    GP_sum = 0
                    if signal == True:
                        success_episode_count = success_episode_count + 1
                now_task = obs['task']
                signal = True 
            
            now_success = obs['success']
            
            #print(signal)
            #print(episode_conut)
            #print(success_episode_count)
            #print("----------------")
            
            HIMA_action,score = agent.get_action(obs)
            if Is_action_success(HIMA_action, obs['teacher_action']):
                obs['score'] = 5
            
            if score >= 4:
                action = HIMA_action
                if obs['score'] >= 4 :
                    interaction_count = interaction_count + 1
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                action = obs['teacher_action']
                if obs['score'] < 4 :
                    interaction_count = interaction_count + 1
                    TN = TN + 1
                else:
                    FN = FN + 1

            check_action = get_action_type(action)
            if check_action == 0:
                signal = False
            print(CLICK_count + SCROLL_count + TYPE_count + PRESS_count + STOP_count)
            print(action)


            print(obs['teacher_action'])
            task = get_action_type(obs['teacher_action'])
            if task == 1:
                CLICK_count = CLICK_count + 1
            elif task == 2:
                SCROLL_count = SCROLL_count + 1    
            elif task == 3:
                TYPE_count = TYPE_count + 1    
            elif task == 4:
                PRESS_count = PRESS_count + 1    
            elif task == 5:
                STOP_count = STOP_count + 1    


            if Is_action_success(action, obs['teacher_action']):
                now_result['success'] = 1
                if task == 1:
                    CLICK_success = CLICK_success + 1
                elif task == 2:
                    SCROLL_success = SCROLL_success + 1    
                elif task == 3:
                    TYPE_success = TYPE_success + 1    
                elif task == 4:
                    PRESS_success = PRESS_success + 1    
                elif task == 5:
                    STOP_success = STOP_success + 1

            else:
                now_result['success'] = 0
                signal = False
            result.append(now_result)
            if Is_action_type(action, obs['teacher_action']):
                total_type = total_type + 1
                if task == 1:
                    CLICK_type = CLICK_type + 1
                if task == 3:
                    TYPE_type = TYPE_type + 1
            now_count = now_count + 1
        except Exception as e:
            print(f"Error processing observation {e}")
            now_count = now_count + 1
            continue  
    
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    CLICK_rate = CLICK_success / CLICK_count
    SCROLL_rate = SCROLL_success / SCROLL_count
    TYPE_rate = TYPE_success / TYPE_count
    PRESS_rate = PRESS_success / PRESS_count
    STOP_rate = STOP_success / STOP_count
    CLICK_type_rate = CLICK_type / CLICK_count
    TYPE_type_rate = TYPE_type / TYPE_count

    TOTAL_rate = (CLICK_success + SCROLL_success + TYPE_success + PRESS_success + STOP_success) / (CLICK_count + SCROLL_count + TYPE_count + PRESS_count + STOP_count)
    TOTAL_type_rate = total_type / (CLICK_count + SCROLL_count + TYPE_count + PRESS_count + STOP_count)

    episode_rate = success_episode_count / episode_conut
    GP = sum(GP_single) / len(GP_single) if GP_single else 0

    interaction_rate = interaction_count / (CLICK_count + SCROLL_count + TYPE_count + PRESS_count + STOP_count)
 
    print(f"CLICK_rate: {CLICK_rate:.4f}")
    print(f"SCROLL_rate: {SCROLL_rate:.4f}")
    print(f"TYPE_rate: {TYPE_rate:.4f}")
    print(f"PRESS_rate: {PRESS_rate:.4f}")
    print(f"STOP_rate: {STOP_rate:.4f}")
    print(f"CLICK_type_rate: {CLICK_type_rate:.4f}")
    print(f"TYPE_type_rate: {TYPE_type_rate:.4f}")
    print(f"TOTAL_rate: {TOTAL_rate:.4f}")
    print(f"TOTAL_type_rate: {TOTAL_type_rate:.4f}")
    print(f"episode_rate: {episode_rate:.4f}")
    print(f"GP: {GP:.4f}")
    print(f"interaction_rate: {interaction_rate:.4f}")
    print(f"TP: {TP:.4f}")
    print(f"FP: {FP:.4f}")
    print(f"TN: {TN:.4f}")
    print(f"FN: {FN:.4f}")

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


    def get_action(self, obs):
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
    And your final goal, previous actions, screen description, action result and associated screenshot are as follows:
    Final goal: {obs['task']}
    Previous actions: {obs['previous_actions']}
    Screenshot: 
    Your output must strictly follow the format below, and especially avoid using unnecessary quotation marks or other punctuation marks.(where osatlas action must be one of the action formats I provided and score must be 1 to 5):
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
                
        action_pattern = r"action:\s*(.*)"
        score_pattern = r"score:\s*(.*)"
        action_match = re.search(action_pattern, output_text[0])
        score_match = re.search(score_pattern, output_text[0])
        if action_match and score_match:
            action = action_match.group(1).strip() 
            try:
                score = score_match.group(1).strip()    
            except:
                score = None
        try:
            score = int(score)  
        except ValueError:  
            score = 1  
        return action, score

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = QwenAgent(device=device, accelerator=accelerator, 
                    policy_lm='/data3/wuzh/aitz_chongzuo/HiMA', max_new_tokens=128)
#Qwen/Qwen2-VL-7B-Instruct
test_path = "/data3/wuzh/HiMA_experiment/main_experiment/HiMA_test.json"

test_loop(agent, test_path)




