
import pandas as pd
import numpy as np

def calculate_simulated_ctr(row, screen_width=1280, screen_height=720):
    """
    Calculates a simulated Click-Through Rate (CTR) based on heuristics.
    """
    ctr_score = 0.0

    # Heuristic 1: Element type bonus
    # Buttons and links are more likely to be clicked.
    if row.get('button_count', 0) > 0 or row.get('link_count', 0) > 0:
        ctr_score += 0.2

    # Heuristic 2: Size bonus (normalized area)
    # Larger elements are more visible.
    area = row.get('avg_area', 0)
    normalized_area = area / (screen_width * screen_height)
    ctr_score += normalized_area * 0.3 # Weight for area

    # Heuristic 3: Center proximity bonus
    # Elements closer to the center are more likely to be seen.
    center_x, center_y = row.get('center_x', 0), row.get('center_y', 0)
    screen_center_x, screen_center_y = screen_width / 2, screen_height / 2
    
    distance_from_center = np.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
    max_distance = np.sqrt(screen_center_x**2 + screen_center_y**2)
    
    # Inverse of distance (proximity)
    proximity_score = 1 - (distance_from_center / max_distance)
    ctr_score += proximity_score * 0.5 # Weight for proximity

    # Apply a sigmoid function to keep the CTR between 0 and 1
    simulated_ctr = 1 / (1 + np.exp(- (ctr_score * 10 - 5))) # Scale and shift to get a nice distribution

    return simulated_ctr

def main():
    """
    Main function to load preprocessed data and generate reward labels.
    """
    input_path = "/home/seokjun/pj/PagePilot/data/preprocessed_waveui.csv"
    output_path = "/home/seokjun/pj/PagePilot/data/labeled_waveui.csv"

    print(f"Loading preprocessed data from {input_path}...")
    df = pd.read_csv(input_path)

    print("Generating simulated CTR labels...")
    df['simulated_ctr'] = df.apply(calculate_simulated_ctr, axis=1)

    print(f"Saving data with labels to {output_path}...")
    df.to_csv(output_path, index=False)

    print("Label generation complete.")
    print("\nSample of the data with the new 'simulated_ctr' column:")
    print(df[['button_count', 'link_count', 'avg_area', 'center_x', 'center_y', 'simulated_ctr']].head())

if __name__ == "__main__":
    main()
