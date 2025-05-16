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

Preprocessing module for the CHB-MIT EEG epilepsy dataset.
"""

import time
import concurrent.futures
from .utils import *
# Configure module logger
logger = logging.getLogger(__name__)


def process_EEG_file(file_path: str, prev_file_path: Optional[str], save_dir: str,
                        annotations_df: pd.DataFrame, channels: List[str],
                        positions: List[List[float]], segment_duration: float,
                        target_duration: float, artifact_epoch_duration: float,
                        ica_components: int, output_epoch_duration: float,
                        overlap: float, interactive: bool = True) -> None:
    """
    Process a single EEG file.

    Args:
        file_path: Path to the EEG file
        prev_file_path: Path to the previous EEG file (or None)
        save_dir: Directory to save processed data
        annotations_df: DataFrame with seizure annotations
        channels: List of channels to use
        positions: Positions of electrodes
        segment_duration: Duration of first segment to cut
        target_duration: Final duration to keep
        artifact_epoch_duration: Duration of epochs for artifact rejection
        ica_components: Number of ICA components to use
        output_epoch_duration: Duration of output epochs
        overlap: Overlap between epochs
        interactive: Whether to use interactive mode
    """
    # Extract filename without directory
    file = os.path.basename(file_path)
    file_str = os.path.splitext(file)[0]  # Remove extension

    # Get all seizure records for this file
    file_seizures = annotations_df[annotations_df.index == file_str]

    if file_seizures.empty:
        logger.info(f"No seizure records found for {file}")
        return

    # Read raw data file (only once)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    logger.info(f'Successfully read {file}')

    # Process each seizure record in this file
    for _, seizure in file_seizures.iterrows():
        seizure_num = seizure['num']
        start_time = seizure['start']

        logger.info(f'Processing seizure {seizure_num} in {file}')

        # Cut segments, add buffer for bad segment removal
        raw_cut = cut_segments(
            raw=raw.copy(),
            prev_file_path=prev_file_path,
            start_time=start_time,
            duration=segment_duration
        )
        logger.info(f'Segment extraction complete for seizure {seizure_num} in {file}')

        # Preprocessing
        raw_copy = raw.copy()
        raw_copy.pick(channels)
        raw_cut.pick(channels)
        logger.info(f'Channel selection complete for {file}')

        # Set montage
        set_montage(raw_copy, channels, positions)
        set_montage(raw_cut, channels, positions)
        logger.info(f'Montage setup complete for {file}')

        # Apply filters
        apply_filter(raw_cut)
        logger.info('Filtering complete')

        if interactive:
            plt.switch_backend('TkAgg')
            plt.ion()
            fig1, (ax1_1, ax1_2) = plt.subplots(2, 1, figsize=(8, 12))
            raw_copy.plot_psd(fmax=100, show=False, ax=ax1_1)
            ax1_1.set_title('Original Data')
            raw_cut.plot_psd(fmax=100, show=False, ax=ax1_2)
            ax1_2.set_title('Filtered Data')
            plt.tight_layout()
            plt.show(block=True)

        # Interpolate bad channels
        interpolate_bad_channels(raw_cut, interactive)
        logger.info(f'First bad channel check complete for {file}')

        # Remove bad segments
        epochs = remove_bad_segments(raw_cut, flag=False, duration=artifact_epoch_duration, interactive=interactive)

        # Remove artifacts and connect each epoch to Raw
        raw_list = []
        turn = 0
        start = time.time()
        for epoch in epochs:
            info = epochs.info
            epoch_raw = mne.io.RawArray(epoch, info)
            epoch_cleaned = remove_artefacts(epoch_raw, n_components=ica_components)
            raw_list.append(epoch_cleaned)
            turn += 1
            logger.debug(f'-----Processed epoch {turn}-----')

        continuous_raw = mne.concatenate_raws(raw_list)
        continuous_raw.set_annotations(None)
        end = time.time()
        logger.info(f'Artifact removal complete for {file}, took {end - start} seconds')

        # Check for remaining artifacts after removal
        if interactive:
            fig3 = raw_cut.plot_psd(fmax=100, show=True)
            fig3.suptitle('Data after artifact removal')
            plt.show(block=True)
            print("Please check if there are any remaining bad channels or segments")

        continuous_raw = remove_bad_segments(
            continuous_raw, flag=True,
            duration=artifact_epoch_duration,
            interactive=interactive
        )
        logger.info(f'Second bad segment removal complete for {file}')

        # Extract target segment
        total_duration = continuous_raw.times[-1]
        logger.debug(f"Total available duration: {total_duration}s")
        if total_duration < target_duration:
            error_msg = f"Insufficient data duration: {total_duration}s available, {target_duration}s required"
            logger.error(error_msg)
            raise ValueError(error_msg)

        raw_target = cut_segments(
            raw=continuous_raw,
            prev_file_path=None,
            start_time=total_duration,
            duration=target_duration
        )
        logger.info(f'Target segment extraction complete for {file}')

        # Compare plots
        if interactive:
            fig3, (ax3_1, ax3_2, ax3_3) = plt.subplots(3, 1, figsize=(10, 12))
            raw_copy.plot_psd(fmax=100, show=False, ax=ax3_1)
            ax3_1.set_title('Original Data')
            raw_cut.plot_psd(fmax=100, show=False, ax=ax3_2)
            ax3_2.set_title('Filtered Data')
            raw_target.plot_psd(fmax=100, show=False, ax=ax3_3)
            ax3_3.set_title('Preprocessed Data')
            plt.tight_layout()
            plt.show(block=True)

        # Save data - modify filename to include seizure event number
        if overlap == 0:
            epochs = mne.make_fixed_length_epochs(
                raw_target,
                reject_by_annotation=False,
                duration=output_epoch_duration,
                preload=True
            )
        else:
            epochs = mne.make_fixed_length_epochs(
                raw_target,
                reject_by_annotation=False,
                duration=output_epoch_duration,
                overlap=overlap,
                preload=True
            )
        epochs_data = epochs.get_data()
        save_path = os.path.join(save_dir, file.replace('.edf', f'_seizure_{int(seizure_num)}.npy'))
        np.save(save_path, epochs_data)
        logger.info(f'Data for seizure {seizure_num} in {file} saved to {save_path}')

    logger.info(f'All seizure records processed for {file}')


def preprocess_data(data_dir: str, save_dir: str, annotations_df: pd.DataFrame,
                    channels: List[str], positions: List[List[float]],
                    segment_duration: float = 3600, target_duration: float = 1800,
                    artifact_epoch_duration: float = 2, ica_components: int = 16,
                    output_epoch_duration: float = 1, overlap: float = 0,
                    interactive: bool = False, parallel: bool = False) -> None:
    """
    Process data files for a single patient, select specified channels and process them,
    then save results as .npy files. Modified to support processing multiple seizure
    records in files.

    Args:
        data_dir: Source data directory
        save_dir: Directory to save data
        annotations_df: DataFrame with seizure times
        channels: List of channels to select (default: None, uses default channel list)
        positions: Channel positions (default: None, uses default positions)
        segment_duration: Duration of first segment to cut (default: 3600 seconds)
        target_duration: Final duration to keep (default: 1800 seconds)
        artifact_epoch_duration: Duration of epochs for artifact detection (default: 2 seconds)
        ica_components: Number of ICA components (default: 16)
        output_epoch_duration: Duration of output epochs (default: 1 second)
        overlap: Overlap between epochs (default: 0)
        interactive: Whether to use interactive plotting (default: True)
        parallel: Whether to process files in parallel (default: False)
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Configure matplotlib for Chinese characters
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Get list of EDF files
    files = extract_file(data_dir, extension=".edf")
    file_paths = [os.path.join(data_dir, file) for file in files]

    # Create list of previous file paths
    prev_file_paths = [None] + file_paths[:-1]

    # Process files
    if parallel and not interactive:
        # Process files in parallel if not in interactive mode
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for file_path, prev_file_path in zip(file_paths, prev_file_paths):
                futures.append(
                    executor.submit(
                        process_EEG_file,
                        file_path=file_path,
                        prev_file_path=prev_file_path,
                        save_dir=save_dir,
                        annotations_df=annotations_df,
                        channels=channels,
                        positions=positions,
                        segment_duration=segment_duration,
                        target_duration=target_duration,
                        artifact_epoch_duration=artifact_epoch_duration,
                        ica_components=ica_components,
                        output_epoch_duration=output_epoch_duration,
                        overlap=overlap,
                        interactive=interactive
                    )
                )
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
    else:
        # Process files sequentially
        for file_path, prev_file_path in zip(file_paths, prev_file_paths):
            try:
                process_EEG_file(
                    file_path=file_path,
                    prev_file_path=prev_file_path,
                    save_dir=save_dir,
                    annotations_df=annotations_df,
                    channels=channels,
                    positions=positions,
                    segment_duration=segment_duration,
                    target_duration=target_duration,
                    artifact_epoch_duration=artifact_epoch_duration,
                    ica_components=ica_components,
                    output_epoch_duration=output_epoch_duration,
                    overlap=overlap,
                    interactive=interactive
                )
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    logger.info(f'Data preprocessing complete for {data_dir}')

