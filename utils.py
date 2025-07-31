import pandas as pd
import numpy as np


def create_aggregated_features(elements_df, all_possible_type_columns=None):
    """
    Creates a pandas DataFrame with aggregated features per screenshot from raw element data.

    Args:
        elements_df: DataFrame containing raw element data for one or more screenshots.
        all_possible_type_columns: A list of all possible one-hot encoded type columns
                                   to ensure consistent feature sets.

    Returns:
        A DataFrame with aggregated features.
    """
    if elements_df.empty:
        # If the input is empty, return a DataFrame with the correct columns but no rows.
        if all_possible_type_columns:
            # Define standard columns that are always present
            standard_cols = [
                'screenshot_id', 'width_mean', 'height_mean', 'area_mean', 'area_sum',
                'center_x_mean', 'center_y_mean', 'text_density_mean', 'component_count'
            ]
            # Combine standard columns with all possible type columns
            all_cols = standard_cols + all_possible_type_columns
            return pd.DataFrame(columns=all_cols)
        else:
            # Fallback if no columns are provided, though this should be avoided
            return pd.DataFrame()

    df = elements_df.copy()

    # Basic feature extraction for each element
    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']
    df['area'] = df['width'] * df['height']
    df['center_x'] = (df['x1'] + df['x2']) / 2
    df['center_y'] = (df['y1'] + df['y2']) / 2
    df['text_density'] = df['ocr_text'].str.len() / (df['area'] + 1e-6)
    df['text_density'] = df['text_density'].fillna(0)

    # One-hot encode the 'type' column
    df_type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, df_type_dummies], axis=1)

    # If a list of all possible type columns is provided, ensure they all exist
    if all_possible_type_columns:
        for col in all_possible_type_columns:
            if col not in df.columns:
                df[col] = 0
        # Ensure the order is correct
        type_columns_to_agg = all_possible_type_columns
    else:
        type_columns_to_agg = [col for col in df.columns if col.startswith('type_')]


    # --- Aggregate features per screenshot_id ---
    # Check if 'screenshot_id' is in the columns to decide on grouping
    if 'screenshot_id' in df.columns:
        agg_dict = {
            'width': 'mean',
            'height': 'mean',
            'area': ['mean', 'sum'],
            'center_x': 'mean',
            'center_y': 'mean',
            'text_density': 'mean',
            **{col: 'sum' for col in type_columns_to_agg}
        }
        agg_df = df.groupby('screenshot_id').agg(agg_dict)
        # Flatten MultiIndex columns
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
        agg_df = agg_df.reset_index()

        # Add a component count feature
        component_counts = df.groupby('screenshot_id').size().reset_index(name='component_count')
        agg_df = pd.merge(agg_df, component_counts, on='screenshot_id')

    else:
        # If no screenshot_id, aggregate the entire DataFrame (for single UI state)
        agg_dict = {
            'width': ['mean'],
            'height': ['mean'],
            'area': ['mean', 'sum'],
            'center_x': ['mean'],
            'center_y': ['mean'],
            'text_density': ['mean'],
            **{col: ['sum'] for col in type_columns_to_agg}
        }
        agg_df = pd.DataFrame(df.agg(agg_dict)).transpose()
        # Flatten MultiIndex columns
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns]
        agg_df['component_count'] = len(df)

    return agg_df
