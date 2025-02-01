import torch
from tqdm import tqdm
import numpy as np
import accelerate
import requests
import json
import re
import PIL.Image
import os
from API import decompose_instruction, Is_single_finished
import time
from ocr_util import get_ocr
import signal
import yaml
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def batch_interact_environment(agent, env, ocr_detection, ocr_recognition,\
        accelerator, post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x, task_id=0):
    
    with open('./config/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)


    try:
        env.terminated = False
        reset_success = False
        env.image_id = str(time.time())
        env.steps = 0
        while not (reset_success):
            for _ in range(5):
                try:
                    if accelerator.is_main_process:
                        with timeout(seconds=480): # change this if frequently timeout
                            env.task_id=task_id
                            print(env.task_id)
                            obs = env.get_obs()
                            print('----------------------')
                            #print(obs)
                        reset_success = True
                    break
                except Exception as e:
                    print(f"Error in environment reset")
                    print(e)
                    accelerate.utils.broadcast(reset_success)
                    continue
        done = False
        
        steps = 0
        if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'):  
            obs['list'] = decompose_instruction(obs['task'])
            obs['now_step'] = 0
        obs['previous_actions'] = []

        while not (done):
            steps += 1
            if accelerator.is_main_process:
                print(f"Environment steps {str(steps)}")
                print("getting actions!")
                
                if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'): 
                    obs['now_step'] = Is_single_finished(obs['list'], obs['image_path'])
                    obs['ocr'] = get_ocr(obs['image_path'], ocr_detection, ocr_recognition)
                
                res = agent.get_action(obs)
                if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and (config['test_mode'] == 'realworld_test' or config['test_mode'] == 'single_test' or config['test_mode'] == 'gpt_test')): 
                    obs['score'] = res['score']
                
                obs['osatlas_action'] = res['osatlas_action']


                if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'): 
                    obs['teacher_action'] = res['teacher_action']
                
                obs['success'] = False
                
                if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'): 
                    if(obs['score']) >= 4:
                        action = obs['osatlas_action']
                    else:
                        action = obs['teacher_action']

                elif config['mode'] == 'test_mode': 
                    if config['test_mode'] == 'the_entire_trajectory':
                        action = obs['osatlas_action']
                    elif config['test_mode'] == 'realworld_test':
                        if(obs['score']) < 4:
                            obs['osatlas_action'] = "HOLD"
                            action = obs['osatlas_action']                                                
                        else:
                            action = obs['osatlas_action']                           
                
                with timeout(seconds=5*60):
                    step_return = env.step(decode_f(action))
                obs_dict, terminate, success = step_return
                
                if obs['osatlas_action'] == "COMPLETE":
                    success = True
                    terminate = True
                
                if success:
                    obs['success'] = True
                
                file_path = config['save_path'] + config['json_name']

                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        try:
                            data = json.load(file)
                        except json.JSONDecodeError:
                            data = [] 
                else:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    data = []
                
                if config['mode'] == 'makedata_mode': 
                    obs_copy = obs.copy()  
                    obs_copy.pop('ocr', None)
                    data.append(obs_copy)
                    print(obs_copy)
                elif config['mode'] == 'test_mode': 
                    data.append(obs)
                    print(obs)
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)
                

                
                if terminate:
                    if success:
                        return 1
                    else:
                        return 0

                if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'): 
                    if(obs['score']) >= 4:
                        obs['previous_actions'].append(obs['osatlas_action'])
                    else:
                        obs['previous_actions'].append(obs['teacher_action'])
                elif config['mode'] == 'test_mode': 
                    obs['previous_actions'].append(obs['osatlas_action'])
                    
                obs["image_path"] = obs_dict["image_path"]
        
        return 0           
            
    except Exception as e:
        print(f"Error in environment interaction")
        import traceback
        print(traceback.format_exc())
        print(e)
        env.terminate()
        return 0

                
        
