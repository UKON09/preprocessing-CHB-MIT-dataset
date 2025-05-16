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

Tools for preprocessing dataset.
"""

import os
import logging
from typing import List, Optional, Union
import mne
from mne.annotations import Annotations
from mne.preprocessing import ICA
from mne_icalabel import label_components
from autoreject import get_rejection_threshold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Configure module logger
logger = logging.getLogger(__name__)


def extract_file(file_dir: str, extension: Optional[str] = None, need_extension: bool = True) -> List[str]:
    """
    Get file names from target path.

    Args:
        file_dir: Target file directory
        extension: Target file extension, e.g. '.edf' (default: None)
        need_extension: Whether to keep the extension in the returned names (default: True)

    Returns:
        List of file names
    """
    files = []
    if extension is not None:
        if need_extension:
            files = [f for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f)) and f.endswith(extension)]
        else:
            files = [os.path.splitext(f)[0] for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f)) and f.endswith(extension)]
    else:
        if need_extension:
            files = [f for f in os.listdir(file_dir)
                     if os.path.isfile(os.path.join(file_dir, f))]
        else:
            raise ValueError("Expect a file extension when need_extension is False.")
    return files


def seizure_annotations(raw: mne.io.Raw, annotations_df: pd.DataFrame, file_str: str) -> None:
    """
    Read .edf files and add seizure annotations.

    Args:
        raw: .edf data file
        annotations_df: Annotation information DataFrame
        file_str: Patient file name without extension
    """
    # Filter rows with index equal to file_str
    filtered_df = annotations_df[annotations_df.index == file_str]
    # Get number of rows
    num = len(filtered_df)
    for i in range(1, num + 1):
        # Parse annotation information
        onsets = filtered_df[(filtered_df['num'] == i)]['start'].values[0]
        durations = filtered_df[(filtered_df['num'] == i)]['duration'].values[0]
        descriptions = 1  # seizures

        # Create Annotations object
        annotations = Annotations(onsets, durations, descriptions)

        # Add annotations to Raw object
        raw.set_annotations(annotations)


def apply_filter(raw: mne.io.Raw) -> None:
    """
    Apply filters to the EEG data.
    Filter design from paper:
    Yikai Yang et al 2023 Neuromorph. Comput. Eng. 3 014010
    - 50 Hz noise: Filter out signals in the 47-53 Hz and 97-103 Hz range.
    - 60 Hz noise: Filter out signals in the 57-63 Hz and 117-123 Hz range.

    Args:
        raw: MNE Raw data object
    """
    # High pass filter
    raw.filter(l_freq=0.1, h_freq=None, method='iir', iir_params=dict(order=5, ftype='butter'))
    # High pass filter
    raw.filter(l_freq=0.5, h_freq=None, method='iir', iir_params=dict(order=6, ftype='butter'))
    # Low pass filter
    raw.filter(l_freq=None, h_freq=70, method='iir', iir_params=dict(order=5, ftype='butter'))
    # 60Hz power line interference
    raw.filter(l_freq=61, h_freq=58, method='fir', fir_design='firwin', phase='zero-double', filter_length='auto',
               l_trans_bandwidth=1, h_trans_bandwidth=1)
    # 50Hz power line interference (commented out)
    # raw.filter(l_freq=51, h_freq=48, method='iir', iir_params=dict(order=1, ftype='butter'))


def cut_segments(raw: mne.io.Raw, prev_file_path: Optional[str], start_time: float,
                 duration: float) -> mne.io.Raw:
    """
    Cut segments based on annotation start time and duration.
    If annotation start time > extraction segment length, cut from (start_time - duration, start_time).
    If annotation start time <= extraction segment length, join with previous segment and cut.

    Args:
        raw: Original data object
        prev_file_path: Path to previous .edf file
        start_time: Start time of the segment to extract
        duration: Duration of the segment to extract

    Returns:
        Cut data segments
    """
    # Initialize segment as None
    segment = None

    # If start_time > duration, cut directly from current segment
    if start_time > duration:
        segment_start = start_time - duration
        segment_end = start_time
        segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
    else:
        # If start_time <= duration, connect current and previous segments
        if prev_file_path is None:
            logger.warning("Previous file path is None, cannot concatenate files")
            segment_start = 0
            segment_end = start_time
            segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
            return segment

        prev_raw = mne.io.read_raw_edf(prev_file_path, preload=True, verbose=False)
        prev_end_time = prev_raw.times[-1]  # End time of previous segment

        if prev_end_time + start_time <= duration:
            prev_segment_start = 0
            prev_segment_end = prev_end_time
            prev_segment = prev_raw.copy().crop(tmin=prev_segment_start, tmax=prev_segment_end)
            segment_start = 0
            segment_end = start_time
            cur_segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
            segment = mne.concatenate_raws([prev_segment, cur_segment])
            logger.warning(f'Previous segment duration insufficient, final segment length: {segment.times[-1]}')
        else:
            prev_segment_start = prev_end_time - duration + start_time
            prev_segment_end = prev_end_time
            prev_segment = prev_raw.copy().crop(tmin=prev_segment_start, tmax=prev_segment_end)
            segment_start = 0
            segment_end = start_time
            cur_segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
            segment = mne.concatenate_raws([prev_segment, cur_segment])

    return segment


def set_montage(raw: mne.io.Raw, channels: List[str], positions: List[List[float]]) -> None:
    """
    Set montage with custom channel positions.
    Map most electrodes to 1020 standard positions, with T7-FT9 and FT10-T8 calculated separately.

    Args:
        raw: MNE Raw data object
        channels: List of bipolar electrode names
        positions: List of 3D positions for each channel
    """
    # Scale positions
    new_positions = [[num / 2000 for num in sub_list] for sub_list in positions]
    channels_positions = dict(zip(channels, new_positions))
    montage = mne.channels.make_dig_montage(ch_pos=channels_positions, coord_frame='head')
    raw.set_montage(montage)


def remove_artefacts(raw: mne.io.Raw, n_components: int = 16, threshold: float = 0.9) -> mne.io.Raw:
    """
    Use ICA and ICLabel to automatically remove artifact components (eye movements, ECG artifacts).

    Args:
        raw: Original EEG data (Raw object)
        n_components: Number of ICA components (default: 16)
        threshold: Probability threshold for determining artifacts (default: 0.9)

    Returns:
        Artifact-free EEG data (Raw object)
    """
    # 1. Run ICA decomposition on EEG data
    raw_copied = raw.copy()
    raw_copied = raw_copied.set_eeg_reference("average")
    ica = ICA(
        n_components=n_components,
        max_iter="auto",
        method="infomax",
        random_state=90,
        fit_params=dict(extended=True),
    )
    ica.fit(raw_copied)
    logger.info('Fitting complete, automatically removing high probability components')

    # Use label_components function to label ICA components
    data = label_components(raw_copied, ica, method='iclabel')
    labels = data['labels']
    probs = data['y_pred_proba']

    # Define list of artifact labels and threshold
    artifact_labels = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']

    # Automatic inspection
    exclude_idx = []
    for i in range(len(labels)):
        label = labels[i]
        prob = probs[i]
        logger.debug(f'Component {i} is {label} with probability {prob}')
        if label in artifact_labels:
            if prob >= threshold:
                exclude_idx.append(i)
                logger.debug(f'Component {i} excluded')
            else:
                logger.debug(f'Component {i} not excluded')

    length = len(exclude_idx)
    if length != 0:
        ica.apply(raw_copied, exclude=exclude_idx)
    logger.info(f"Excluded {length} components")

    return raw_copied


def interpolate_bad_channels(raw: mne.io.Raw, interactive: bool = True) -> None:
    """
    Plot EEG and interpolate bad channels.

    Args:
        raw: MNE Raw data object
        interactive: Whether to use interactive mode
    """
    if interactive:
        raw.plot(block=True, title='Please check for bad channels')
        user_input = input(
            "Enter channel names to mark as bad, separated by spaces (e.g., MEG 0111 EEG 001): ")
        bads = [channel.strip() for channel in user_input.split() if channel.strip()]
    else:


        # Create epochs for auto detection
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
        reject = get_rejection_threshold(epochs)

        # Find channels that exceed thresholds frequently
        bad_channel_counts = {}
        for ch_type, thresh in reject.items():
            if ch_type == 'eeg':
                data = epochs.get_data(picks='eeg')
                for i, ch_name in enumerate(epochs.ch_names):
                    if ch_name not in bad_channel_counts:
                        bad_channel_counts[ch_name] = 0
                    # Count how many times channel exceeds threshold
                    ch_data = data[:, i, :]
                    if ch_type == 'eeg':
                        peak_to_peak = np.ptp(ch_data, axis=1)
                        bad_count = np.sum(peak_to_peak > thresh)
                        bad_channel_counts[ch_name] += bad_count

        # Channels with high bad counts are marked as bad
        threshold = 0.3 * epochs.get_data().shape[0]  # 30% of epochs
        bads = [ch for ch, count in bad_channel_counts.items() if count > threshold]
        logger.info(f"Auto-detected bad channels: {bads}")

    raw.info['bads'] = bads
    if len(raw.info['bads']) != 0:
        raw.interpolate_bads()
        logger.info('Bad channel interpolation complete')
    else:
        logger.info('No bad channels, interpolation not needed')


def remove_bad_segments(raw: mne.io.Raw, flag: bool, duration: float = 2.0,
                        interactive: bool = True) -> Union[mne.io.Raw, mne.Epochs]:
    """
    Remove bad segments.

    Args:
        raw: MNE Raw data object
        flag: Whether to return continuous epochs (True) or not (False)
        duration: Epoch duration (default: 2.0 seconds)
        interactive: Whether to use interactive mode

    Returns:
        If flag is True: raw (continuous epochs)
        If flag is False: epochs
    """
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True)

    if interactive:
        fig = epochs.plot(show=True, title='Please select bad segments')
        plt.show(block=True)
        logger.info('Bad segment removal complete')
    else:
        reject = get_rejection_threshold(epochs)
        epochs.drop_bad(reject=reject)
        logger.info(f'Auto-rejected {len(epochs.drop_log)} bad segments')

    if flag:
        raw_list = []
        for epoch in epochs:
            info = epochs.info
            epoch_raw = mne.io.RawArray(epoch, info)
            raw_list.append(epoch_raw)
        continuous_raw = mne.concatenate_raws(raw_list)
        return continuous_raw
    else:
        return epochs