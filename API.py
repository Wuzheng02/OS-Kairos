import os
#import google.generativeai as genai
import PIL.Image
import re
import yaml
from openai import OpenAI
import base64
from zhipuai import ZhipuAI

with open('./config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)



def get_teacher_action(final_goal, current_goal, image_path, ocr, step_list, previous_actions):
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
    
    teacher_model = config['teacher_model']
    '''
    if teacher_model == 'gemini':
        gemini_key = config['gemini_key']
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        screenshot = PIL.Image.open(image_path)
        response = model.generate_content([prompt,screenshot])
        answer = response.text
    '''
    if teacher_model == 'gpt':
        client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key= config['openai_key']
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
          #model = "qwen-vl-max",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
    
    elif teacher_model == 'glm':
        client = ZhipuAI(
            api_key= config['glm_key']
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
                                    "url":base64_image
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content


    start = answer.find('action:') + 8
    end = answer.rfind('}')
    action = answer[start:end]
    print(action)
    return action
    

def get_teacher_score(final_goal, current_goal, image_path, osatlas_action, ocr, teacher_action, previous_actions):  
    prompt = (
        "### Background ###\n"
        "You are an expert in completing tasks based on screenshots and instructions. "
        "You will grade the student action based on the goal, screenshot, and teacher action. I hope you can be a bit stricter in your scoring."
        "I will provide you with a mobile screenshot, a final goal, the current goal, the previous actions,a student action and a teacher action. "
        "I hope you evaluate this student action based on the screenshot , the teacher action and the goal, giving it a score from 1 to 5. "
        "The teacher action is an example you consider worthy of a full score (5 points). If you believe the student action does not achieve the same level of performance, points should be deducted accordingly. Pay special attention to cases involving coordinates; significant discrepancies in coordinates must result in point deductions."
        f"Final goal：{final_goal}\n"
        f"current goal：{current_goal}\n"
        f"student action:{osatlas_action}\n"
        f"previous actions : {previous_actions}"
        f"teacher action:{teacher_action}\n"
        "### Screenshot information ###\n"
        "To help you understand the information in the screenshot, I first performed OCR. Here are the names and coordinates of the icons obtained through OCR:"
        f"Coordinates of the icons: {ocr}"
        "### Response requirements ###\n"
        "I hope you evaluate this action based on the screenshot and the goal, giving it a score from 1 to 5. "
        "A higher score indicates that you believe this action is more likely to accomplish the current goal for the given screenshot.\n"
        "1 means you believe this action definitely cannot achieve the goal.\n"
        "2 means you believe this action is very unlikely to achieve the goal.\n"
        "3 means you believe this action has a certain chance of achieving the goal.\n"
        "4 means you believe this action is very likely to achieve the goal.\n"
        "5 means you believe this action will definitely achieve the goal.\n"
        "If the teacher action and student action are of different types, the score should only be between 1 and 3 points."
        "If both the teacher action and student action are CLICK, a full score of 5 points can be given if the coordinate difference is minimal. However, if the coordinate difference is significant, points must be deducted."
        "### Output format ###\n"
        "Your output must strictly follow the format below:\n"
        "{score: }"
    )
    teacher_model = config['teacher_model']
    '''
    if teacher_model == 'gemini':
        gemini_key = config['gemini_key']
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        screenshot = PIL.Image.open(image_path)  
        response = model.generate_content([prompt, screenshot])
        answer = response.text
    '''    
    if teacher_model == 'gpt':
        client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key= config['openai_key']
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
                                    "url":f"data:image/png;base64,{base64_image}"
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
    
    elif teacher_model == 'glm':
        client = ZhipuAI(
            api_key= config['glm_key']
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
                                    "url":base64_image
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content

    start = answer.find('score:') + 7
    end = answer.rfind('}')
    result = answer[start:end]
    score = int(result)
    print(score)
    return score

def decompose_instruction(goal):
    prompt = (
        "你现在是一个手机软件使用专家。我需要你帮我把一条操作手机软件的指令分解成多阶段的分步指令，请严格按照我的示例格式输出"
        "例如：\n"
        "原指令：去高德地图查看到上海交通大学的路线\n"
        "分解指令：(1)打开高德地图\n"
        "         (2)点击首页的搜索框\n"
        "         (3)输入：上海交通大学\n"
        "         (4)点击搜索结果\n"
        "         (5)点击屏幕右下角的路线按钮\n"
        "         (6)指令完成\n"
        "原指令：打开高德地图，查看附近的厕所\n"
        "分解指令：(1)打开高德地图\n"
        "         (2)点击首页下方的附近按钮\n"
        "         (3)点击寻找和厕所有关的信息并点进去\n"
        "         (4)指令完成\n"   
        f"原指令: {goal}\n"
        "分解指令:\n"
        "请直接输出分解指令:"       
    )

    teacher_model = config['teacher_model']
    '''
    if teacher_model == 'gemini':
        gemini_key = config['gemini_key']
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        answer = response.text
    '''
    if teacher_model == 'gpt':
        client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key= config['openai_key']
        )
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
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
    elif teacher_model == 'glm':
        client = ZhipuAI(
            api_key= config['glm_key']
        )
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt,
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content

    print(answer)
    instructions = re.findall(r"\(\d+\)(.*?)\n", answer)
    return instructions


def Is_single_finished(step_list, image_path):
    prompt = (
        "### Background ###\n"
        "You are an expert in completing tasks based on screenshots and instructions. "
        "I will provide you with a mobile screenshot and a step list. "
        "You should be able to tell from the screenshot and the list of steps what step you are currently at. "
        f"The step list is: {step_list}\n"
        "### Response requirements ###\n"
        "You can only output the index of a list of steps."
        "For example, step_list: ["
        "Open WeChat, "
        "Click the Contacts or Search button (depending on your version of WeChat and settings), "
        "If you click on Contacts, find and click on your wife's avatar; if you click on the Search button, enter your wife's name or note in the search box, "
        "Go to your and your wife's WeChat, "
        "Go to the chat screen between you and your wife, "
        "Click on the input box, "
        "Enter: I'm HIMA the Intelligence, I'm going home for the weekend tonight, no more studying, thanks!, "
        "Click the send button"
        "], if you think you're still in the main screen, output {step index: 0}; if you've completed the task 'Open WeChat', you output {step index: 1}; if you think you've completed clicking on the input box, you output {step index: 5}.. Your output should just be a number."
        "That means that the output is produced after a certain number of steps have been completed."
        "### Output format ###\n"
        "Your output must strictly follow the format below:\n"
        "{step index: }"
    )
    
    teacher_model = config['teacher_model']
    '''
    if teacher_model == 'gemini':
        gemini_key = config['gemini_key']
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        screenshot = PIL.Image.open(image_path)  
        response = model.generate_content([prompt, screenshot])
        answer = response.text
    '''
    if teacher_model == 'gpt':
        client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key= config['openai_key']
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
                                    "url":f"data:image/png;base64,{base64_image}"
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
    elif teacher_model == 'glm':
        client = ZhipuAI(
            api_key= config['glm_key']
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
                                    "url":base64_image
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
    start = answer.find(':') + 2
    end = answer.rfind('}')
    result = answer[start:end]
    index = int(result)
    return index

def Is_final_finished(previous_screenshot, current_screenshot, current_task): 
    prompt = (
        "You are an expert in completing tasks based on screenshots and instructions. "
        "I am now providing you with a screenshot of the previous state, a screenshot of the current state, and the overall goal."
        "You should be able to tell from the screenshot and the list of steps what step you are currently at. "
        f"The overall goal is: {current_task}\n"
        "Please determine whether the overall goal has been achieved based on the overall goal and the screenshots. If you believe it has been achieved, output 1. If you believe it has not been achieved, output 0.\n"
        "Your output must strictly follow the format below:\n"
        "{Is_final_finished: 0} or {Is_final_finished: 1}"
    )
    teacher_model = config['teacher_model']
    '''
    if teacher_model == 'gemini':
        gemini_key = config['gemini_key']
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        screenshot1 = PIL.Image.open(previous_screenshot)
        screenshot2 = PIL.Image.open(current_screenshot)  
        response = model.generate_content([prompt, screenshot1, screenshot2])
        answer = response.text
    '''
    if teacher_model == 'gpt':
        client = OpenAI(
            base_url="https://api.gpts.vin/v1",
            api_key= config['openai_key']
        )
        with open(previous_screenshot, "rb") as image_file:
            base64_image1 = base64.b64encode(image_file.read()).decode('utf-8')
        with open(current_screenshot, "rb") as image_file:
            base64_image2 = base64.b64encode(image_file.read()).decode('utf-8')
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
                                    "url":f"data:image/png;base64,{base64_image1}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":f"data:image/png;base64,{base64_image2}"
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
    
    elif teacher_model == 'glm':
        client = ZhipuAI(
            api_key= config['glm_key']
        )
        with open(previous_screenshot, "rb") as image_file:
            base64_image1 = base64.b64encode(image_file.read()).decode('utf-8')
        with open(current_screenshot, "rb") as image_file:
            base64_image2 = base64.b64encode(image_file.read()).decode('utf-8')
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
                                    "url":base64_image1
                                    },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":base64_image2
                                    },
                            },
                        ],
                    }
                ]
        completion = client.chat.completions.create(
          model="glm-4v-plus",
          messages=messages
        )
        chat_response = completion
        answer = chat_response.choices[0].message.content
    #print(answer)
    start = answer.find('Is_final_finished:') + 19
    end = answer.rfind('}')
    result = answer[start:end]
    print(result)
    Is_final_finished = int(result)
    return Is_final_finished
