"""
Unit tests for EEG preprocessing tools using pytest
"""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import mne
from mne.annotations import Annotations
from mne.channels import make_standard_montage
from edfio import Edf, EdfSignal
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend to avoid showing plots during tests

from preprocessing_CHB_MIT_dataset import utils
from preprocessing_CHB_MIT_dataset.utils import (
    extract_file, seizure_annotations, cut_segments, set_montage,
    apply_filter, interpolate_bad_channels, remove_artefacts, remove_bad_segments
)
from preprocessing_CHB_MIT_dataset.advanced_preprocess import (
    process_EEG_file, preprocess_data
)

# Set logging to warning level to reduce test output
import logging

logging.basicConfig(level=logging.WARNING)


# Common fixtures for all tests
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def test_files(temp_dir):
    """Create test files in the temporary directory."""
    files = ['test1.edf', 'test2.edf', 'test3.txt']
    for file in files:
        with open(os.path.join(temp_dir, file), 'w') as f:
            f.write('test content')
    return files


@pytest.fixture
def raw_data():
    """Create a test Raw object."""
    data = np.random.randn(5, 1000)  # 5 channels, 1000 sample points
    info = mne.create_info(['Fp1', 'Fp2', 'Fpz', 'Fz', 'FCz'],
                           sfreq=100, ch_types='eeg')
    return mne.io.RawArray(data, info)


@pytest.fixture
def annotations_df():
    """Create test annotations dataframe."""
    df = pd.DataFrame({
        'num': [1, 1, 2],
        'start': [50, 20, 100],
        'duration': [10, 20, 30]
    })
    df.index = ['test1', 'test2', 'test2']
    return df


@pytest.fixture
def raw_with_montage(raw_data):
    """Create a Raw object with montage."""
    raw = raw_data.copy()
    montage = make_standard_montage('standard_1020')
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
    raw.set_montage(montage)
    return raw


@pytest.fixture
def channels_and_positions():
    """Create channels and positions for montage testing."""
    channels = ['CH1', 'CH2', 'CH3']
    positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    return channels, positions


@pytest.fixture
def save_dir():
    """Create a temporary directory for saving output files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


# Tests for utils.py
class TestTools:
    """Test functions in the utils module."""

    def test_extract_file(self, temp_dir, test_files):
        """Test extract_file function."""
        # Test extracting .edf files
        edf_files = extract_file(temp_dir, extension='.edf')
        assert len(edf_files) == 2
        assert 'test1.edf' in edf_files
        assert 'test2.edf' in edf_files

        # Test extracting without extensions
        edf_files_no_ext = extract_file(temp_dir, extension='.edf', need_extension=False)
        assert len(edf_files_no_ext) == 2
        assert 'test1' in edf_files_no_ext
        assert 'test2' in edf_files_no_ext

        # Test extracting all files
        all_files = extract_file(temp_dir)
        assert len(all_files) == 3

    def test_seizure_annotations(self, raw_data, annotations_df):
        """Test seizure_annotations function."""
        # Apply annotations
        seizure_annotations(raw_data, annotations_df, 'test1')

        # Verify annotations were added
        annotations = raw_data.annotations
        assert len(annotations) == 1  # Only the first annotation should be added
        assert annotations.onset[0] == 50
        assert annotations.duration[0] == 10
        assert annotations.description[0] == '1'

    def test_cut_segments_direct_cut(self, raw_data):
        """Test cut_segments function with direct cut (start_time > duration)."""
        start_time = 80
        duration = 30
        segment = cut_segments(raw_data, None, start_time, duration)

        # Verify cut result
        assert segment.times[-1] == 30  # Should have 30 seconds of data

    def test_set_montage(self, channels_and_positions):
        """Test set_montage function."""
        channels, positions = channels_and_positions

        # Create test Raw
        data = np.random.randn(3, 1000)
        info = mne.create_info(channels, sfreq=100, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # Apply montage
        set_montage(raw, channels, positions)

        # Verify montage was set
        assert raw.info['dig'] is not None


# Tests for signal processing functions
class TestSignalProcessing:
    """Test signal processing functions."""

    def test_apply_filter(self, raw_with_montage):
        """Test apply_filter function."""
        # Save original spectrum
        psd_original, freqs = raw_with_montage.compute_psd(fmax=70).get_data(return_freqs=True)

        # Apply filter
        apply_filter(raw_with_montage)

        # Calculate filtered spectrum
        psd_filtered, _ = raw_with_montage.compute_psd(fmax=70).get_data(return_freqs=True)

        # Verify low and high frequency components are attenuated
        # Low frequency (< 0.5 Hz)
        low_freq_idx = np.where(freqs < 0.5)[0]
        if len(low_freq_idx) > 0:
            assert np.mean(psd_filtered[:, low_freq_idx]) < np.mean(psd_original[:, low_freq_idx])

        # High frequency (> 60 Hz)
        high_freq_idx = np.where(freqs > 60)[0]
        if len(high_freq_idx) > 0:
            assert np.mean(psd_filtered[:, high_freq_idx]) < np.mean(psd_original[:, high_freq_idx])

    def test_interpolate_bad_channels(self, raw_with_montage):
        """Test interpolate_bad_channels function."""
        # Set non-interactive mode for testing
        # Simulate a bad channel
        raw_with_montage.info['bads'] = ['CH1']

        # Call function
        interpolate_bad_channels(raw_with_montage, interactive=False)

        # Verify bad channel was interpolated (check raw_with_montage.info['bads'] is empty)
        assert len(raw_with_montage.info['bads']) == 0

    def test_remove_artefacts(self, raw_with_montage):
        """Test remove_artefacts function."""
        # Prepare data with artificial artifacts
        data = raw_with_montage.get_data()
        # Add simulated eye movement artifact to CH1
        blink_indices = np.arange(1000, 1200)
        data[0, blink_indices] = 5 * np.sin(np.linspace(0, np.pi, len(blink_indices)))

        # Create new Raw object
        raw_with_artifacts = mne.io.RawArray(data, raw_with_montage.info.copy())

        # Apply filter (ICA typically needs filtered data)
        apply_filter(raw_with_artifacts)

        # Call function - Note: In real environment this needs MNE_ICALABEL training data
        # Here we just test that the function call doesn't error
        try:
            cleaned_raw = remove_artefacts(raw_with_artifacts, n_components=3, threshold=0.8)
            # Verify return value is a Raw object
            assert isinstance(cleaned_raw, mne.io.Raw)
        except ImportError:
            # Skip test if mne_icalabel is not available
            pytest.skip("mne_icalabel not available")

    def test_remove_bad_segments(self, raw_with_montage):
        """Test remove_bad_segments function."""
        # Prepare data with bad segments
        data = raw_with_montage.get_data()
        # Inject abnormal values at certain time points
        data[:, 500:520] = 100  # Abnormally large values

        # Create new Raw object
        raw_with_bad = mne.io.RawArray(data, raw_with_montage.info.copy())

        # Call function, in non-interactive mode returning epochs
        epochs = remove_bad_segments(raw_with_bad, flag=False, interactive=False)

        # Verify return is epochs object
        assert isinstance(epochs, mne.Epochs)

        # Call function, in non-interactive mode returning continuous data
        continuous = remove_bad_segments(raw_with_bad, flag=True, interactive=False)

        # Verify return is Raw object
        assert isinstance(continuous, mne.io.Raw)


# Tests for advanced preprocessing functions
class TestAdvancedPreprocess:
    """Test advanced preprocessing functions."""

    @pytest.fixture
    def test_eeg_setup(self, temp_dir):
        """Set up test EEG data."""
        # Create test data
        # 1. Build simple edf file
        fs = 100
        data = np.random.randn(5, 100*fs)  # 5 channels, 10000 sample points (100s data)
        channels = ['CH1', 'CH2', 'CH3', 'CH4', 'CH5']
        signals = []
        for i in range(5):
            data = data[:, i]
            channel = channels[i]
            signal = EdfSignal(
                data=data,
                label=channel,
                sampling_frequency=fs,
                transducer_type="Ag/AgCl",
                physical_dimension="uV"
            )
            signals.append(signal)

        # 2. Save as edf file
        edf = Edf(
            signals=signals,
        )
        test_file_path = os.path.join(temp_dir, 'test_file.edf')
        edf.write(test_file_path)

        # 3. Create annotations dataframe
        annotations_df = pd.DataFrame({
            'num': [1],
            'start': [50],
            'duration': [40]
        })
        annotations_df.index = ['test_file']

        # 4. Define channels and positions
        positions = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]]

        return {
            'file_path': test_file_path,
            'annotations_df': annotations_df,
            'channels': channels,
            'positions': positions
        }

    def test_process_EEG_file(self, test_eeg_setup, save_dir):
        """Test process_EEG_file function."""
        # Since this function is complex, we only test that it doesn't error
        try:
            # Use non-interactive mode
            process_EEG_file(
                file_path=test_eeg_setup['file_path'],
                prev_file_path=None,
                save_dir=save_dir,
                annotations_df=test_eeg_setup['annotations_df'],
                channels=test_eeg_setup['channels'],
                positions=test_eeg_setup['positions'],
                segment_duration=30,  # Shorter to speed up testing
                target_duration=20,
                artifact_epoch_duration=1,
                ica_components=3,
                output_epoch_duration=1,
                overlap=0,
                interactive=False
            )
            # Check if output file was generated
            output_file = os.path.join(save_dir, 'test_file_seizure_1.npy')
            assert os.path.exists(output_file)
        except Exception as e:
            # Fail the test if there's an exception
            pytest.fail(f"process_EEG_file raised an exception: {str(e)}")

    def test_preprocess_data(self, test_eeg_setup, temp_dir, save_dir):
        """Test preprocess_data function."""
        # Since this function calls process_EEG_file, we only do a simple error test
        try:
            # Use non-interactive and non-parallel mode
            preprocess_data(
                data_dir=temp_dir,
                save_dir=save_dir,
                annotations_df=test_eeg_setup['annotations_df'],
                channels=test_eeg_setup['channels'],
                positions=test_eeg_setup['positions'],
                segment_duration=30,  # Shorter to speed up testing
                target_duration=20,
                artifact_epoch_duration=1,
                ica_components=3,
                output_epoch_duration=1,
                overlap=0,
                interactive=False,
                parallel=False
            )
            # Check if output file was generated
            output_file = os.path.join(save_dir, 'test_file_seizure_1.npy')
            assert os.path.exists(output_file)
        except Exception as e:
            # Fail the test if there's an exception
            pytest.fail(f"preprocess_data raised an exception: {str(e)}")


# Helper functions for manual testing
def generate_test_data():
    """Generate simulated data set for manual testing."""
    # Create test directories
    os.makedirs('test_data', exist_ok=True)
    os.makedirs('test_output', exist_ok=True)

    # 1. Create simulated EEG data
    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3']
    sfreq = 256  # Sampling rate
    duration = 300  # 5 minutes of data
    n_channels = len(channels)
    n_times = int(duration * sfreq)

    # Baseline signal
    data = np.random.randn(n_channels, n_times) * 10  # Microvolt level

    # Add Alpha wave activity (8-12 Hz)
    t = np.arange(n_times) / sfreq
    alpha_freq = 10  # Hz
    alpha_amp = 15
    alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
    data[2, :] += alpha  # Add Alpha activity to one channel

    # Add muscle artifacts (>20 Hz)
    muscle_start = int(100 * sfreq)
    muscle_end = int(110 * sfreq)
    muscle = np.random.randn(muscle_end - muscle_start) * 50
    data[0, muscle_start:muscle_end] += muscle

    # Add eye movement artifacts
    blink_starts = [int(50 * sfreq), int(150 * sfreq), int(250 * sfreq)]
    for start in blink_starts:
        blink_len = int(0.5 * sfreq)  # 0.5s blink
        blink = 100 * np.sin(np.linspace(0, np.pi, blink_len))
        data[0, start:start + blink_len] += blink

    # Add epileptic seizure-like pattern
    seizure_start = int(200 * sfreq)
    seizure_end = int(220 * sfreq)
    t_seizure = np.arange(seizure_end - seizure_start) / sfreq
    seizure_freq = 3  # Hz, similar to slow wave sleep or epileptic EEG
    seizure = 80 * np.sin(2 * np.pi * seizure_freq * t_seizure)
    for ch in range(n_channels):
        data[ch, seizure_start:seizure_end] += seizure

    # Create MNE Raw object
    info = mne.create_info(channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Add epileptic seizure annotations
    onset = seizure_start / sfreq  # seconds
    duration = (seizure_end - seizure_start) / sfreq  # seconds
    annotations = Annotations(onset=[onset], duration=[duration], description=['1'])
    raw.set_annotations(annotations)

    # Save as EDF file
    edf_path = os.path.join('test_data', 'test_seizure.edf')
    raw.save(edf_path, overwrite=True)

    # 2. Create test annotation file
    annotations_df = pd.DataFrame({
        'num': [1],
        'start': [onset],
        'duration': [duration]
    })
    annotations_df.index = ['test_seizure']
    csv_path = os.path.join('test_data', 'annotations.csv')
    annotations_df.to_csv(csv_path)

    # 3. Create electrode position file
    positions = [
        [0, 0, 0],  # FP1-F7
        [1, 0, 0],  # F7-T7
        [2, 0, 0],  # T7-P7
        [3, 0, 0],  # P7-O1
        [0, 1, 0]  # FP1-F3
    ]
    positions_df = pd.DataFrame(positions, index=channels)
    pos_path = os.path.join('test_data', 'positions.csv')
    positions_df.to_csv(pos_path)

    print(f"Test data generated in 'test_data' directory")
    print(f"EDF file: {edf_path}")
    print(f"Annotations file: {csv_path}")
    print(f"Positions file: {pos_path}")

    return channels, positions, annotations_df


@pytest.mark.manual
def test_manual_run():
    """Run manual test (marked to be skipped in automated testing)."""
    # Generate test data
    channels, positions, annotations_df = generate_test_data()

    # Run preprocessing
    preprocess_data(
        data_dir='test_data',
        save_dir='test_output',
        annotations_df=annotations_df,
        channels=channels,
        positions=positions,
        segment_duration=60,  # 1 minute, shortened to speed up testing
        target_duration=40,  # 40 seconds
        artifact_epoch_duration=1,
        ica_components=4,  # Fewer components to speed up running
        output_epoch_duration=1,
        overlap=0,
        interactive=True,  # Interactive mode
        parallel=False
    )

    print("Preprocessing complete, results saved in 'test_output' directory")