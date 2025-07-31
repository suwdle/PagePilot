import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rl_env import PagePilotEnv
from dqn_trainer import DQN


def render_ui_from_elements(elements_list, canvas_size=(800, 600)):
    """
    Renders a UI layout from a list of element dictionaries and returns it as a PIL Image.
    """
    img = Image.new('RGB', canvas_size, color='white')
    draw = ImageDraw.Draw(img)

    try:
        # Try to use a common font, fallback to default
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for element in elements_list:
        x1, y1, x2, y2 = element['x1'], element['y1'], element['x2'], element['y2']
        element_type = element.get('type', 'unknown')
        ocr_text = element.get('ocr_text', '')

        # Define colors based on element type
        color_map = {
            "button": ("blue", "lightblue"),
            "link": ("purple", "plum"),
            "image": ("green", "lightgreen"),
            "text": ("black", "whitesmoke"),
            "default": ("gray", "lightgray")
        }
        outline_color, fill_color = color_map.get(element_type, color_map["default"])

        draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color)

        # Add text (OCR or type)
        display_text = ocr_text if ocr_text else element_type
        # Truncate long text to fit within the box
        if len(display_text) > 50:
            display_text = display_text[:47] + "..."
        draw.text((x1 + 5, y1 + 5), display_text, fill="black", font=font)

    return img


def run_optimization_visualizer(env, policy_net, steps=10):
    """
    Runs the DQN agent for a few steps and yields the state for visualization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset env and get initial state
    initial_state_vector = env.reset()
    initial_reward = env._get_reward()
    yield "initial", env.state, initial_reward

    state = torch.tensor(initial_state_vector, dtype=torch.float32, device=device).unsqueeze(0)

    for step in range(steps):
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)

        # Apply action to the environment's state
        element_index = action.item() // 4
        move_direction = action.item() % 4

        if element_index < len(env.state):
            env._take_action(element_index, move_direction)
        else:
            # If action is invalid, we can skip or just yield the same state
            yield "no_change", env.state, env._get_reward()
            continue

        optimized_reward = env._get_reward()
        yield f"step_{step+1}", env.state, optimized_reward

        # Update state for the next iteration
        state_vector = env._state_to_vector(env.state)
        state = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
