import pandas as pd
from datasets import load_dataset

def vectorize_waveui_sample(sample):
    """
    Vectorizes a single WaveUI sample using statistics-based fixed vector (Method B).
    
    Args:
        sample (dict): A single sample from the WaveUI dataset.
        
    Returns:
        list: A fixed-size vector representing the UI layout.
    """
    
    # Initialize features
    num_elements = 0
    avg_width = 0
    avg_height = 0
    element_type_counts = {}

    # Process bbox data
    individual_bboxes = []
    if 'bbox' in sample and isinstance(sample['bbox'], list):
        # Assuming bbox is a flat list of coordinates [x1,y1,x2,y2, x3,y3,x4,y4, ...]
        # We need to group them into individual bboxes
        flat_bbox_list = sample['bbox']
        for i in range(0, len(flat_bbox_list), 4):
            if i + 3 < len(flat_bbox_list): # Ensure there are 4 coordinates
                individual_bboxes.append(flat_bbox_list[i:i+4])
            else:
                print(f"Warning: Incomplete bbox coordinates at end: {flat_bbox_list[i:]}")

        num_elements = len(individual_bboxes)
        if num_elements > 0:
            bbox_widths = []
            bbox_heights = []
            for bbox in individual_bboxes:
                bbox_widths.append(bbox[2] - bbox[0])
                bbox_heights.append(bbox[3] - bbox[1])
            avg_width = sum(bbox_widths) / num_elements
            avg_height = sum(bbox_heights) / num_elements

    # Process element types
    if 'type' in sample and isinstance(sample['type'], list):
        for elem_type in sample['type']:
            element_type_counts[elem_type] = element_type_counts.get(elem_type, 0) + 1

    # For simplicity, let's just return a few features for now.
    vector = [
        num_elements,
        avg_width,
        avg_height,
        element_type_counts.get('button', 0), # Number of buttons
        element_type_counts.get('link', 0),   # Number of links
        element_type_counts.get('text', 0)    # Number of text elements
    ]

    return vector

def main():
    print("Starting WaveUI Vectorization...")
    
    try:
        # Load a small subset of the WaveUI dataset for demonstration
        # ds = load_dataset("agentsea/wave-ui-25k", split='train[:10]') # Load first 10 samples
        # For local testing, if dataset is already downloaded, you might use:
        ds = load_dataset("agentsea/wave-ui-25k", split='train')
        
        print(f"Loaded {len(ds)} samples from WaveUI-25K dataset.")
        
        # Vectorize a few samples
        num_samples_to_vectorize = min(5, len(ds))
        print(f"Vectorizing {num_samples_to_vectorize} samples...")
        
        for i in range(num_samples_to_vectorize):
            sample = ds[i]
            vector = vectorize_waveui_sample(sample)
            print(f"Sample {i} vectorized: {vector}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
