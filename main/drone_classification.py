import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import scipy.io.wavfile as sci_wav
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from all_mfcc_functions.mfcc_extraction import extract_mfcc_feature
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import time

# Define data set root paths
#fs = 16000  # 16kHz sampling rate
fs = 44100  # 44.1kHz sampling rate

# Use absolute paths for reliability
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR_BACKGROUND_TRAIN = os.path.join(ROOT_DIR, 'data', 'train', 'background') + os.sep
ROOT_DIR_DRONE_TRAIN = os.path.join(ROOT_DIR, 'data', 'train', 'drone') + os.sep
ROOT_DIR_HELICOPTER_TRAIN = os.path.join(ROOT_DIR, 'data', 'train', 'helicopter') + os.sep

ROOT_DIR_BACKGROUND_TEST = os.path.join(ROOT_DIR, 'data', 'test', 'background') + os.sep
ROOT_DIR_DRONE_TEST = os.path.join(ROOT_DIR, 'data', 'test', 'drone') + os.sep
ROOT_DIR_HELICOPTER_TEST = os.path.join(ROOT_DIR, 'data', 'test', 'helicopter') + os.sep

def read_wav_files(root_dir: str, wav_files: list):
    """
    Read audio data from provided paths.

    :param root_dir: Path to root directory
    :param wav_files: List of .wav files to read
    :return: Audio data
    """
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    
    audio_data = []
    for f in wav_files:
        file_path = os.path.join(root_dir, f)
        try:
            # Read the audio file
            sample_rate, data = sci_wav.read(file_path)
            
            # Convert stereo to mono if needed
            if len(data.shape) > 1 and data.shape[1] > 1:
                print(f"Converting stereo to mono for {f}")
                data = np.mean(data, axis=1).astype(data.dtype)
            
            audio_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return audio_data


def read_data():
    """
    Read background, drone and helicopter data from train and test folders

    :return: Train and test data samples with labels
    """
    # List all audio files
    background_train_files = [f for f in os.listdir(ROOT_DIR_BACKGROUND_TRAIN) if f.lower().endswith('.wav')]
    drone_train_files = [f for f in os.listdir(ROOT_DIR_DRONE_TRAIN) if f.lower().endswith('.wav')]
    helicopter_train_files = [f for f in os.listdir(ROOT_DIR_HELICOPTER_TRAIN) if f.lower().endswith('.wav')]
    
    background_test_files = [f for f in os.listdir(ROOT_DIR_BACKGROUND_TEST) if f.lower().endswith('.wav')]
    drone_test_files = [f for f in os.listdir(ROOT_DIR_DRONE_TEST) if f.lower().endswith('.wav')]
    helicopter_test_files = [f for f in os.listdir(ROOT_DIR_HELICOPTER_TEST) if f.lower().endswith('.wav')]
    
    # Read audio files
    data_background_train = read_wav_files(ROOT_DIR_BACKGROUND_TRAIN, background_train_files)
    data_drone_train = read_wav_files(ROOT_DIR_DRONE_TRAIN, drone_train_files)
    data_helicopter_train = read_wav_files(ROOT_DIR_HELICOPTER_TRAIN, helicopter_train_files)
    
    data_background_test = read_wav_files(ROOT_DIR_BACKGROUND_TEST, background_test_files)
    data_drone_test = read_wav_files(ROOT_DIR_DRONE_TEST, drone_test_files)
    data_helicopter_test = read_wav_files(ROOT_DIR_HELICOPTER_TEST, helicopter_test_files)
    
    # Store filenames for debugging and analysis
    all_test_files = []
    all_test_files.extend([("background", f) for f in background_test_files])
    all_test_files.extend([("drone", f) for f in drone_test_files])
    all_test_files.extend([("helicopter", f) for f in helicopter_test_files])
    
    # Create datasets and labels (0 = background, 1 = drone, 2 = helicopter)
    ds_train = data_background_train + data_drone_train + data_helicopter_train
    labels_train = np.concatenate((
        np.zeros(len(data_background_train)),
        np.ones(len(data_drone_train)),
        np.full(len(data_helicopter_train), 2)
    ))
    
    ds_test = data_background_test + data_drone_test + data_helicopter_test
    labels_test = np.concatenate((
        np.zeros(len(data_background_test)),
        np.ones(len(data_drone_test)),
        np.full(len(data_helicopter_test), 2)
    ))
    
    print(f"Training set: {len(ds_train)} samples")
    print(f"  - Background: {len(data_background_train)} samples")
    print(f"  - Drone: {len(data_drone_train)} samples")
    print(f"  - Helicopter: {len(data_helicopter_train)} samples")
    
    print(f"Test set: {len(ds_test)} samples")
    print(f"  - Background: {len(data_background_test)} samples")
    print(f"  - Drone: {len(data_drone_test)} samples")
    print(f"  - Helicopter: {len(data_helicopter_test)} samples")
    
    return ds_train, labels_train, ds_test, labels_test, all_test_files


def pad_data(data_set: list, fix_length: int):
    """
    Pad each sample from data set to fix length

    :param data_set: Input data
    :param fix_length: Fix length in number of samples to pad data
    :return: Data set with fix length
    """
    if not isinstance(data_set, list):
        return librosa.util.fix_length(data_set, size=fix_length, axis=0, mode='wrap')
    else:
        data_set_fix_length = np.zeros((len(data_set), fix_length))
        for i, data in enumerate(data_set):
            data_set_fix_length[i, :] = librosa.util.fix_length(data, size=fix_length, axis=0, mode='wrap')
    return data_set_fix_length


def mfcc_extraction(data_set_fix_length: list, fs: float, n_fft: int, frame_size: float, frame_step: float):
    """
    Extract MFCC features for each data sample.

    :param data_set_fix_length: Input data set
    :param fs: Sampling frequency
    :param n_fft: Num of Nfft points
    :param frame_size: Size of frame in sec
    :param frame_step: Frame step in sec
    :return: MFCC features for each data sample
    """
    flag = True
    mfcc_features_result = None
    
    for i, data in enumerate(data_set_fix_length):
        mfcc = extract_mfcc_feature(y=data, fs=fs, n_fft=n_fft, frame_size=frame_size, frame_step=frame_step)
        if flag:
            mfcc_features_result = np.zeros((len(data_set_fix_length), mfcc.shape[0], mfcc.shape[1]))
            flag = False
        mfcc_features_result[i, :, :] = mfcc
    
    return mfcc_features_result


def preprocess_raw_data(data: list, fix_length: int, mfcc_parameters: dict):
    """
    Preprocess audio data. Returns MFCC features of each audio sample.

    :param data: List of audio data
    :param fix_length: Fix length to pad each audio sample
    :param mfcc_parameters: Parameters for MFCC extraction
    :return: MFCC data for each audio sample
    """
    # pad data to fix length
    data_set_fix_length = pad_data(data, fix_length)
    
    # extract MFCC for each sample
    mfcc_features = mfcc_extraction(
        data_set_fix_length, 
        mfcc_parameters['fs'], 
        mfcc_parameters['n_fft'],
        mfcc_parameters['frame_size'], 
        mfcc_parameters['frame_step']
    )

    return mfcc_features


def show_confusion_matrix(title: str, confusion_matrix, class_names: list, save_path=None, normalize=False):
    """
    Generates plot of confusion matrix and optionally saves it to a file.

    :param title: Plot title
    :param confusion_matrix: Confusion matrix
    :param class_names: List of class names
    :param save_path: Optional path to save the plot as PNG
    :param normalize: Whether to normalize the confusion matrix (True) or show raw counts (False)
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        # Normalize confusion matrix
        cm_display = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        cm_display = np.round(cm_display, 2)
        fmt = ".2f"
        subtitle = " (Normalized)"
    else:
        # Use raw counts
        cm_display = confusion_matrix
        fmt = "d"
        subtitle = " (Counts)"
    
    plt.title(title + subtitle, fontsize=16)
    
    # Create dataframe
    df_cm = pd.DataFrame(cm_display, index=class_names, columns=class_names)

    # Plot heatmap
    cmap = "Blues" if normalize else "YlGnBu"
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap=cmap, 
                          square=True, linewidths=.5)

    # Adjust labels
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show(block=False)


def show_confusion_matrix_comparison(title: str, cm, class_names: list, save_path=None):
    """
    Generates a side-by-side plot of normalized and raw count confusion matrices.

    :param title: Plot title
    :param cm: Confusion matrix
    :param class_names: List of class names
    :param save_path: Optional path to save the plot as PNG
    """
    # Create a wide figure for side-by-side plots
    plt.figure(figsize=(18, 8))
    plt.suptitle(title, fontsize=18, y=0.98)
    
    # Create normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm, 2)
    
    # Left plot - Normalized
    plt.subplot(1, 2, 1)
    plt.title("Normalized (Ratio)", fontsize=14)
    df_cm_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    sns.heatmap(df_cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                square=True, linewidths=.5)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    # Right plot - Raw counts
    plt.subplot(1, 2, 2)
    plt.title("Raw Counts", fontsize=14)
    df_cm_raw = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm_raw, annot=True, fmt="d", cmap="YlGnBu", 
                square=True, linewidths=.5)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
    plt.tight_layout()
    
    # Save the combined figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved side-by-side confusion matrix to {save_path}")
    
    plt.show(block=False)


if __name__ == '__main__':
    # Load data
    data_set_train, y_train, data_set_test, y_test, test_files = read_data()

    # Set parameters
    n_fft = 512
    mfcc_parameters = {
        'fs': fs,
        'n_fft': n_fft,
        'frame_size': n_fft / fs,
        'frame_step': int(n_fft / 3) / fs,
    }

    # Pick the fixed length in seconds
    fix_len_s = 5.0  # 5.0s (adjust based on your audio durations)
    fix_len = int(fix_len_s * fs)

    # Extract MFCC features
    X_train = preprocess_raw_data(data_set_train, fix_len, mfcc_parameters)
    X_train = np.array([feature.ravel() for feature in X_train])
    X_test = preprocess_raw_data(data_set_test, fix_len, mfcc_parameters)
    X_test = np.array([feature.ravel() for feature in X_test])

    # Use PCA for dimensionality reduction
    n_components = 40  # Increased from 30 to capture more variance for 3 classes
    pca = PCA(n_components=n_components)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Print explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")

    # Train logistic regression with multi-class support and stronger regularization
    # logreg = LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial', max_iter=1000) # Needed replacing with line below.
    logreg = LogisticRegression(C=10, solver='lbfgs', max_iter=1000)

    # Fit model and make predictions
    logreg.fit(X_train, y_train)
    y_train_pred = logreg.predict(X_train)
    y_test_pred = logreg.predict(X_test)

    # Print reports
    print('\nTrain classification report:')
    print(classification_report(y_train, y_train_pred, 
                              target_names=['background', 'drone', 'helicopter']))
    
    print('\nTest classification report:')
    print(classification_report(y_test, y_test_pred, 
                              target_names=['background', 'drone', 'helicopter']))
    
    # Print individual test file predictions
    class_names = ['background', 'drone', 'helicopter']
    print("\nIndividual test file predictions:")
    print("---------------------------------")
    
    misclassified_count = 0
    
    for i, (true_label, prediction) in enumerate(zip(y_test, y_test_pred)):
        true_class = class_names[int(true_label)]
        predicted_class = class_names[int(prediction)]
        file_class, file_name = test_files[i]
        
        # Check if prediction was correct and format output accordingly
        if true_label == prediction:
            result = "CORRECT"
        else:
            result = "WRONG"
            misclassified_count += 1
            
        print(f"{result}: {file_class}/{file_name} → Predicted: {predicted_class}")
    
    print(f"\nSummary: {misclassified_count} out of {len(y_test)} files misclassified ({misclassified_count/len(y_test)*100:.1f}%)")

    # Create output directory for saving plots if it doesn't exist
    output_dir = os.path.join(ROOT_DIR, "drone_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Generate and save confusion matrices
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Individual matrices (normalized and raw counts)
    train_cm_norm_path = os.path.join(output_dir, f"{timestamp}_train_confusion_matrix_normalized.png")
    test_cm_norm_path = os.path.join(output_dir, f"{timestamp}_test_confusion_matrix_normalized.png")
    train_cm_counts_path = os.path.join(output_dir, f"{timestamp}_train_confusion_matrix_counts.png")
    test_cm_counts_path = os.path.join(output_dir, f"{timestamp}_test_confusion_matrix_counts.png")
    
    # Side-by-side comparison matrices
    train_comparison_path = os.path.join(output_dir, f"{timestamp}_train_confusion_matrix_comparison.png")
    test_comparison_path = os.path.join(output_dir, f"{timestamp}_test_confusion_matrix_comparison.png")
    
    # Calculate confusion matrices once
    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Show and save normalized matrices
    show_confusion_matrix(title="Train data", confusion_matrix=train_cm, class_names=class_names, 
                         save_path=train_cm_norm_path, normalize=True)
    show_confusion_matrix(title="Test data", confusion_matrix=test_cm, class_names=class_names, 
                         save_path=test_cm_norm_path, normalize=True)
    
    # Show and save raw count matrices
    show_confusion_matrix(title="Train data", confusion_matrix=train_cm, class_names=class_names, 
                         save_path=train_cm_counts_path, normalize=False)
    show_confusion_matrix(title="Test data", confusion_matrix=test_cm, class_names=class_names, 
                         save_path=test_cm_counts_path, normalize=False)
    
    # Show and save side-by-side comparison matrices
    show_confusion_matrix_comparison(title="Train Data - Confusion Matrix", cm=train_cm, 
                                   class_names=class_names, save_path=train_comparison_path)
    show_confusion_matrix_comparison(title="Test Data - Confusion Matrix", cm=test_cm, 
                                   class_names=class_names, save_path=test_comparison_path)
    
    print(f"\nConfusion matrix images saved to {output_dir} directory")
    
    plt.show()

"""
                    | Predicted Class
                    | Background | Drone | Helicopter
--------------------|------------|-------|------------
         Background |     A      |   B   |     C
Actual    Drone     |     D      |   E   |     F
Class     Helicopter|     G      |   H   |     I

A: Correctly classified background noise
E: Correctly classified drone noise
I: Correctly classified helicopter noise
B: Background noise incorrectly classified as drone noise
C: Background noise incorrectly classified as helicopter noise
D: Drone noise incorrectly classified as background noise
F: Drone noise incorrectly classified as helicopter noise
G: Helicopter noise incorrectly classified as background noise
H: Helicopter noise incorrectly classified as drone noise
"""