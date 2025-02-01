from env_util import batch_interact_environment
import numpy as np
from misc import colorful_print
import os
import torch
import time
import json
from action_util import qwen_translate_action, ActionType
import math
def the_entire_trajectory_loop(env,\
                agent,\
                accelerator,\
                tokenizer,\
                ocr_detection,\
                ocr_recognition,\
                eval_nums: int = 10,\
                use_wandb: bool = False,\
                save_path: str = None,\
                decode_f: callable = lambda x: x,\
                **kwargs):
    agent.prepare()
    done_nums = 0
    for i in range(eval_nums):
        done_nums = done_nums + batch_interact_environment(agent = agent,\
                                                       env = env,\
                                                       accelerator = accelerator,\
                                                       use_tqdm=False,\
                                                       decode_f = decode_f,\
                                                       ocr_detection=ocr_detection,\
                                                       ocr_recognition=ocr_recognition,\
                                                       task_id=i)

    
    successful_rate = done_nums / eval_nums
    print("successful_rate:")
    print(successful_rate)
 

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

def the_single_test_loop(agent, test_path):
    with open(test_path, 'r') as file:
        data = json.load(file)

    element_count = 0

    just_osatlas_success_count = 0
    success_count = 0

    interaction_count = 0

    just_osatlas_step_type_true_count = 0
    step_type_true_count = 0


    for obs in data:
        try:

            osatlas_action, score = agent.get_osaction_and_score(obs)
            score = int(score)
            element_count = element_count + 1
            print(osatlas_action)
            print(obs['teacher_action'])

            if Is_action_success(osatlas_action, obs['teacher_action']):
                just_osatlas_success_count = just_osatlas_success_count + 1
                obs['score'] = 5
            if Is_action_type(osatlas_action, obs['teacher_action']):
                just_osatlas_step_type_true_count = just_osatlas_step_type_true_count + 1
            
            if score < 4:
                action = obs['teacher_action']
                if obs['score'] < 4:
                    interaction_count = interaction_count + 1
            else:
                action = osatlas_action
                if obs['score'] >= 4:
                    interaction_count = interaction_count + 1
            
            if Is_action_success(action, obs['teacher_action']):
                success_count = success_count + 1
            if Is_action_type(action, obs['teacher_action']):
                step_type_true_count = step_type_true_count + 1

            print(element_count)
            print(just_osatlas_success_count)
            print(success_count)
            print(interaction_count)
            #print(step_type_true_count)
        except Exception as e:
            print(f"Error processing observation {element_count}: {e}")
            continue  

    just_osatlas_success_rate = just_osatlas_success_count / element_count
    single_step_success_rate = success_count / element_count

    interaction_rate = interaction_count / element_count

    just_osatlas_type_true_rate = just_osatlas_step_type_true_count / element_count
    single_type_true_rate = step_type_true_count / element_count
    

    print(just_osatlas_success_rate)
    print(single_step_success_rate)

    print(interaction_rate)

    print(just_osatlas_type_true_rate)
    print(single_type_true_rate)