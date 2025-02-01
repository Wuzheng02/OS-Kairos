from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
from PIL import Image


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


def to_autoui(act: AndroidAction):
    if act.action_type == ActionType.DualPoint:
        return f'"action_type": "DUAL_POINT", "touch_point": "[{act.touch_point[1]:.4f}, {act.touch_point[0]:.4f}]", "lift_point": "[{act.lift_point[1]:.4f}, {act.lift_point[0]:.4f}]", "typed_text": ""'
    elif act.action_type == ActionType.Type:
        return f'"action_type": "TYPE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": "{act.typed_text}"'
    elif act.action_type == ActionType.GoBack:
        return f'"action_type": "PRESS_BACK", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.GoHome:
        return f'"action_type": "PRESS_HOME", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.Enter:
        return f'"action_type": "PRESS_ENTER", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    elif act.action_type == ActionType.TaskComplete or act.action_type == ActionType.TaskImpossible:
        return f'"action_type": "STATUS_TASK_COMPLETE", "touch_point": "[-1.0, -1.0]", "lift_point": "[-1.0, -1.0]", "typed_text": ""'
    else:
        print(f"Action {act} not supported yet.")
        return ""


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