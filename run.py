import os
import transformers
from env import AndroidEmulator
from qwen_agent import QwenAgent
from loop import the_entire_trajectory_loop, the_single_test_loop
from misc import colorful_print
from action_util import qwen_translate_action
import torch.nn as nn
import numpy as np 
from omegaconf import DictConfig, OmegaConf
import yaml
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()
import torch
import torch.distributed as dist
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import math

def load_task_file(assets_path):
    all_tasks = []
    with open(os.path.join(assets_path, "instructions.txt")) as fb: 
        for line in fb:
            all_tasks.append(line.replace("\n", ""))
    return all_tasks


def main():
    with open('./config/config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(minutes=40)), kwargs_handlers=[ddp_kwargs], project_dir = config['save_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = None

    
    all_tasks = load_task_file(config['assets_path'])
    translate_action = qwen_translate_action
    use_feature_extractor = False


    decode_f = lambda x:x

    if config['mode'] == 'makedata_mode':
        agent = QwenAgent(device=device, accelerator=accelerator, 
                      policy_lm=config['policy_lm'], max_new_tokens=config['max_new_tokens'])
    elif config['mode'] == 'test_mode':
        agent = QwenAgent(device=device, accelerator=accelerator, 
                      policy_lm=config['sft_lm'], max_new_tokens=config['max_new_tokens'])
    tokenizer = agent.tokenizer    


    def construct_env(sample_mode):
        env = AndroidEmulator(
            max_steps=config['max_steps']-1, # will have 1 dangling step after stop signal is triggered
            translate_action=translate_action,
            temp_path = os.path.join(config['save_path'], "images"),
            save_images=True,
            all_tasks=all_tasks,
            sample_mode=sample_mode,
            task_id=0,
        )
        return env


    env = construct_env(sample_mode=config['eval_sample_mode'])
    if config['mode'] == 'makedata_mode' or (config['mode'] == 'test_mode' and config['test_mode'] == 'gpt_test'):
        ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')
        the_entire_trajectory_loop(env = env,
                tokenizer=tokenizer,
                agent = agent,
                accelerator = accelerator,
                decode_f=decode_f,
                ocr_detection=ocr_detection,
                ocr_recognition=ocr_recognition,
                **config)
    elif config['mode'] == 'test_mode':
        if config['test_mode'] == 'the_entire_trajectory' or config['test_mode'] == 'realworld_test':
            the_entire_trajectory_loop(env = env,
                tokenizer=tokenizer,
                agent = agent,
                accelerator = accelerator,
                decode_f=decode_f,
                ocr_detection=None,
                ocr_recognition=None,
                **config)                       
        elif config['test_mode'] == 'single_step':
            the_single_test_loop(agent = agent, test_path = config['test_path'])
            
if __name__ == "__main__":
    main()
