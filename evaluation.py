import torch
import pandas as pd
import numpy as np
from rl_env import PagePilotEnv
from dqn_trainer import DQN
from PIL import Image, ImageDraw, ImageFont

def render_ui_from_elements(elements_list, file_path, canvas_size=(800, 600)):
    """
    Renders a UI layout from a list of element dictionaries and saves it as an image.
    """
    img = Image.new('RGB', canvas_size, color = 'white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for element in elements_list:
        x1, y1, x2, y2 = element['x1'], element['y1'], element['x2'], element['y2']
        element_type = element['type']
        ocr_text = element['ocr_text']

        # Draw rectangle
        color = "blue" if element_type == "button" else "gray"
        fill_color = "lightblue" if element_type == "button" else "lightgray"
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill_color)

        # Add text (OCR or type)
        display_text = ocr_text if ocr_text else element_type
        draw.text((x1 + 5, y1 + 5), display_text, fill="black", font=font)
        
    img.save(file_path)


def evaluate_agent(env, policy_net):
    """
    Evaluates the DQN agent and visualizes the optimization process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select a random screenshot_id for visualization
    initial_screenshot_id = np.random.choice(env.unique_screenshot_ids)
    env.current_screenshot_id = initial_screenshot_id
    env.state = env.raw_data[env.raw_data['screenshot_id'] == initial_screenshot_id].to_dict('records')

    # --- Visualization Setup ---
    images = []
    output_gif_path = "./models/optimization_process.gif"

    print(f"--- Visualizing Optimization for Screenshot ID: {initial_screenshot_id} ---")

    # 1. Initial State
    initial_reward = env._get_reward()
    print(f"Step 0: Initial Predicted CTR: {initial_reward:.4f}")
    
    # Render initial UI
    render_ui_from_elements(env.state, f"./models/step_0.png")
    images.append(Image.open(f"./models/step_0.png"))
    
    # 2. Optimization Loop
    state_vector = env._state_to_vector(env.state)
    state = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)
    
    for step in range(10): # 10 optimization steps
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
        
        # Apply action to the environment's state
        element_index = action.item() // 4
        move_direction = action.item() % 4
        if element_index < len(env.state):
            env._take_action(element_index, move_direction)
        
        optimized_reward = env._get_reward()
        print(f"  Step {step+1}: Action={action.item()}, New CTR={optimized_reward:.4f}")
        
        # Render the new UI state
        render_ui_from_elements(env.state, f"./models/step_{step+1}.png")
        images.append(Image.open(f"./models/step_{step+1}.png"))

        # Update state for the next iteration
        state_vector = env._state_to_vector(env.state)
        state = torch.tensor(state_vector, dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Save as GIF
    images[0].save(output_gif_path,
                   save_all=True, append_images=images[1:],
                   optimize=False, duration=500, loop=0)
                   
    print(f"\nVisualization complete. GIF saved to {output_gif_path}")


def main():
    """
    Main function to load the model and run the evaluation.
    """
    # Load environment and data
    data_path = "./data/raw_elements.csv" # Use raw data
    reward_model_path = "./models/reward_simulator_lgbm.joblib"
    env = PagePilotEnv(data_path, reward_model_path)

    # Load the trained DQN model
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN(input_dim, output_dim).to(device)
    policy_net.load_state_dict(torch.load("./models/dqn_model.pth", map_location=device))
    policy_net.eval()

    # Run the evaluation and visualization
    evaluate_agent(env, policy_net)

if __name__ == "__main__":
    main()
