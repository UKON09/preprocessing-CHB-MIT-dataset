"""
Copyright (c) 2025 UKON09. All rights reserved.
This software is released under the MIT License.
See LICENSE in the project root for details.
Project Name: preprocess_CHB-MIT_dataset
Author: UKON09
Project Version: v1.2.0
Python Version: 3.10+
Created: 2025/5/15
GitHub: https://github.com/UKON09/preprocessing-CHB-MIT-dataset

Main entry point for preprocessing the CHB-MIT EEG epilepsy dataset.
Uses configuration from config.toml file.
"""

import os
import logging
import tomli
from typing import Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
from .preprocess_data import preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to configuration file (default: config.toml)

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def main():
    """Main function to run the preprocessing pipeline."""
    # Load configuration
    config = load_config()

    # Extract configuration parameters
    data_dir = config["paths"]["data_dir"]
    output_dir = config["paths"]["output_dir"]
    annotations_path = config["paths"]["annotations_path"]

    # Get preprocessing parameters
    channels = config["preprocessing"]["picked_channels"]
    positions = config["preprocessing"]["channel_positions"]

    # Get segment durations
    first_segment_duration = config["preprocessing"]["first_segment_duration"]
    second_segment_duration = config["preprocessing"]["second_segment_duration"]

    # Get ICA settings
    artifact_epoch_duration = config["preprocessing"]["artifact_epoch_duration"]
    ica_components = config["preprocessing"]["ICA_components"]

    # Get output settings
    output_epoch_duration = config["preprocessing"]["output_epoch_duration"]
    overlap = config["preprocessing"]["overlap_present"]

    # Get interactive settings
    interactive = config["preprocessing"]["interactive"]

    # Get parallel process setting
    parallel = config["preprocessing"]["parallel"]

    # Create output directory for preictal data
    preictal_dir = os.path.join(output_dir, "preictal")
    os.makedirs(preictal_dir, exist_ok=True)

    # Configure visualization
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Load annotations
    # annotation table structure：
    # file（index）     num       start       end       duration
    # chb06_01          1         1724       1738        14
    # chb06_01          2         1800       1838        38
    try:
        annotations_df = pd.read_csv(annotations_path, index_col="file")
        logger.info(f"Loaded annotations for {len(annotations_df)} files")
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return

    # Process all patients
    with os.scandir(data_dir) as entries:
        for entry in entries:
            if entry.is_dir():
                print(f"成功进入{entry.name}")
                preprocess_data(
                    data_dir=entry.path,
                    save_dir=preictal_dir,
                    annotations_df=annotations_df,
                    channels=channels,
                    positions=positions,
                    segment_duration=first_segment_duration,
                    target_duration=second_segment_duration,
                    artifact_epoch_duration=artifact_epoch_duration,
                    ica_components=ica_components,
                    output_epoch_duration=output_epoch_duration,
                    overlap=overlap,
                    interactive=interactive,
                    parallel=parallel  # Set to True for non-interactive parallel processing
                )


if __name__ == "__main__":
    main()