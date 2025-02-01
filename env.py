import os
import subprocess, signal
import re
from time import sleep
import random
from action_util import ActionType, ActionType, AndroidAction
import time
from misc import colorful_print
import base64
from PIL import Image
from io import BytesIO
from termcolor import colored, cprint
import concurrent.futures
import numpy as np
import traceback
from API import Is_final_finished
import yaml

def adb_screenshot(temp_path, name):
   
    screenshot_cmd = (
        f"ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        f"'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell screencap -p /sdcard/{name} && adb pull /sdcard/{name} ./{name}\"'"
    )
    
    screenshot_cmd2 = (
        f"sshpass -p '811737662' scp -i ~/.ssh/id_rsa -o ProxyJump='wuzh@60.165.238.180:9530' -P 2222 "
        f"wuzh@localhost:/C:/Users/wuzh/{name} {temp_path}/"
    )
    sleep(4)
    subprocess.run(screenshot_cmd, shell=True)
    subprocess.run(screenshot_cmd2, shell=True)

def adb_input_text(text):


    base64_text = base64.b64encode(text.encode('utf-8'))
    base64_text = base64_text.decode('utf-8')
    input_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        f"'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell am broadcast -a ADB_INPUT_B64 --es msg {base64_text}\"'"
    ) 


    """
    formatted_text = text.replace(" ", "%s")
    input_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        f"'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input text {formatted_text}\"'"
    )
    """
    subprocess.run(input_cmd, shell=True)


def adb_back():
    back_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        "'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input keyevent 4\"'"
    )
    subprocess.run(back_cmd, shell=True)

def adb_go_home():
    go_home_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        "'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input keyevent 3\"'"
    )
    subprocess.run(go_home_cmd, shell=True)

def adb_enter():
    enter_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        "'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input keyevent 66\"'"
    )
    subprocess.run(enter_cmd, shell=True)

def adb_tap(x, y):
    tap_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        "'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input tap {0} {1}\"'".format(x, y)
    )
    subprocess.run(tap_cmd, shell=True)

def adb_swipe(x1, y1, x2, y2):
    swipe_cmd = (
        "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
        "'sshpass -p \"811737662\" ssh -p 2222 wuzh@localhost \"adb shell input touchscreen swipe {0} {1} {2} {3}\"'".format(x1, y1, x2, y2)
    )
    subprocess.run(swipe_cmd, shell=True)


def escape_shell_text(text):
    chars_to_escape = ['\\','"', "'", '`', '$']
    for char in chars_to_escape:
        text = text.replace(char, '\\' + char)
    text = text.replace(" ", "%s")
    return text

class AndroidEmulator():
    def __init__(self, max_steps, temp_path, all_tasks = None, translate_action = None, save_images = False, task_id=0, sample_mode=None):
        """
        temp_path temporary path to store the images for evaluation
        """
        self.temp_path = temp_path
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        self.save_images = save_images
        self.image_id = str(time.time())
        self.terminated = False
        self.max_steps = max_steps
        self.steps = 0
        self.task_id = 0
        self.all_tasks = all_tasks
        if sample_mode == "random":
            # randomly sample a task from the task set
            self.current_task = random.choice(all_tasks)
        elif sample_mode == "sequential":
            self.current_task = all_tasks[self.task_id]
        else:
            print("Invalid sample mode")
        self.translate_action = translate_action
        self.history = []

    
    def terminate(self):
        list_packages_cmd = "ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 'sshpass -p 811737662 ssh -p 2222 wuzh@localhost \"adb shell pm list packages -3\"'"
        packages_output = subprocess.check_output(list_packages_cmd, shell=True).decode('utf-8')

        for package in packages_output.strip().split('\n'):
            package_name = package.split(':')[1]
            stop_cmd = (
                f"ssh -i ~/.ssh/id_rsa -p 9530 wuzh@60.165.238.180 "
                f"'sshpass -p 811737662 ssh -p 2222 wuzh@localhost \"adb shell am force-stop {package_name}\"'"
            )
            subprocess.run(stop_cmd, shell=True)
        adb_go_home()

    def count_white_pixels(self, img):
        img = img.convert('RGB')
        data = np.array(img)
        white_count = np.sum(np.all(data > 240, axis=-1))
        return white_count > 2_300_000
    
    def get_obs(self):
        for _ in range(3):
            try:
                self.current_task = self.all_tasks[self.task_id]
                is_white = True
                imagepath = os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png")
                name=f"{self.image_id}_{self.steps}.png"
                adb_screenshot(self.temp_path,name)
                return {
                        "task": self.current_task,
                        "image_path": imagepath,
                }          
            except Exception as e:
                print(f"Exception happened during screenshotting")
                print(e)
                print(traceback.format_exc())
                sleep(6)
                continue

    def step(self, raw_action: str):
        
        with open('./config/config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if raw_action == "HOLD":
            self.steps += 1
            
            if self.steps > self.max_steps:
                cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
                terminated = True
            input("HiMA requires human-machine interaction, where users complete an action and press the Enter key to continue.")
            screenshot = self.get_obs()
            success = Is_final_finished(os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png"), os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"), self.current_task)
            if success:
                terminated = True
            return screenshot, self.terminated, success
        
        self.current_task = self.all_tasks[self.task_id]
        if self.terminated:
            return None
        try:
            action = self.translate_action(raw_action)
        except Exception as e:
            print(e)
            print(f"Failed to translate action: {raw_action}, terminating the environment")
            action = AndroidAction(action_type=ActionType.TaskImpossible)
        self.history.append(action)
        self.steps += 1
        if self.steps > self.max_steps:
            action = AndroidAction(action_type=ActionType.TaskImpossible)
            cprint(colored(f"Terminate the Emulator: Max Steps Exceeded {self.max_steps}.", "red"))
        screenshot = None
        info = {}

        for i in range(2):
            try:
                if action.action_type == ActionType.DualPoint:
                    assert len(action.touch_point) == 2
                    assert len(action.lift_point) == 2
                    if(action.touch_point!=action.lift_point):
                        adb_swipe(action.touch_point[0]*1084, action.touch_point[1]*2412, action.lift_point[0]*1084, action.lift_point[1]*2412)
                    else:
                        adb_tap(action.touch_point[0]*1084, action.touch_point[1]*2412)
                elif action.action_type == ActionType.Type:
                    adb_input_text(action.typed_text)
                elif action.action_type == ActionType.GoBack:
                    adb_back()
                elif action.action_type == ActionType.GoHome:
                    adb_go_home()
                elif action.action_type == ActionType.Enter:
                    adb_enter()
                elif action.action_type == ActionType.TaskComplete:
                    self.terminated = True
                elif action.action_type == ActionType.TaskImpossible:
                    self.terminated = True
                elif action.action_type == ActionType.Idle:
                    pass
                else:
                    raise Exception(f"Unknown action type: {action.action_type}")
                action_success = True
                screenshot = self.get_obs()
                break
            except Exception as e:
                cprint(colored("an Exception occurred during environment interaction", "red"))
                print(e)
                cprint(colored("Retrying", "red"))
                sleep(10)
                if i == 1:
                    action_success = False
                    info["error"] = str(e)
                    self.terminate()
                    return None
                continue
        
        if config['mode'] == "makedata_mode" or config['test_mode'] != "realworld_test":
            success = Is_final_finished(os.path.join(self.temp_path, f"{self.image_id}_{self.steps-1}.png"), os.path.join(self.temp_path, f"{self.image_id}_{self.steps}.png"), self.current_task)
        else:
            if action.action_type == ActionType.TaskComplete:
                success = True
            else:
                success = False

        if success:
            self.terminated = True
        if self.terminated:
            self.terminate()
        return screenshot, self.terminated, success



