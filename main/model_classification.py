import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import numpy as np
import scipy.io.wavfile as sci_wav
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from all_mfcc_functions.mfcc_extraction import extract_mfcc_feature
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import time

# Define drone models to classify
DRONE_MODELS = [
    "X4_D1_MATRICE300", 
    "X4_D2_MAVIC3", 
    "X4_D3_MavicAir2S", 
    "X4_D4_MavicAir2", 
    "X4_D5_MavicMini2", 
    "X4_D6_Mavic2Pro",
    "X4_D7_Mavic2Pro",
    "X4_D8_Mavic3",
    "X4_D9_Phantom4", 
    "X4_D10_Mavic2Zoom",
    "X4_D11_MavicMini2",
    "X4_D17_Phantom4",
    "X6_D12_YuneecH520", 
    "X6_D13_YuneecH520ERTK", 
    "X6_D14_S900", 
    "X6_D15_X6D", 
    "X6_D16_Y6"
]

# Define data paths
fs = 16000  # 16kHz sampling rate
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Base directories for train and test sets
TRAIN_DIR = os.path.join(ROOT_DIR, 'data', 'train')
TEST_DIR = os.path.join(ROOT_DIR, 'data', 'test')

# Print the directories we're looking in
print(f"Looking for drone model folders in:\n  Training: {TRAIN_DIR}\n  Testing: {TEST_DIR}")

# Check if directories exist
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Training directory does not exist: {TRAIN_DIR}")
    
if not os.path.exists(TEST_DIR):
    print(f"ERROR: Testing directory does not exist: {TEST_DIR}")

# List all subdirectories in train and test to help debug
if os.path.exists(TRAIN_DIR):
    print("\nAvailable subdirectories in training directory:")
    train_subdirs = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    for d in train_subdirs:
        print(f"  - {d}")
        
if os.path.exists(TEST_DIR):
    print("\nAvailable subdirectories in testing directory:")
    test_subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    for d in test_subdirs:
        print(f"  - {d}")


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
    Read train and test data for each drone model

    :return: Train and test data samples with labels and file info
    """
    train_data = []
    train_labels = []
    train_files = []
    
    test_data = []
    test_labels = []
    test_files = []
    
    # Loop through each drone model
    for idx, model in enumerate(DRONE_MODELS):
        # Train data
        model_train_dir = os.path.join(TRAIN_DIR, model)
        print(f"Looking for training files in {model_train_dir}")
        
        if os.path.exists(model_train_dir):
            # Look for .wav or .WAV files
            model_train_files = [f for f in os.listdir(model_train_dir) 
                               if f.lower().endswith('.wav') and not f.startswith('.')]
                               
            if model_train_files:
                print(f"Found {len(model_train_files)} WAV files for {model} in training directory")
                
                for file in model_train_files:
                    try:
                        # Read the audio file
                        file_path = os.path.join(model_train_dir, file)
                        sample_rate, data = sci_wav.read(file_path)
                        
                        # Convert stereo to mono if needed
                        if len(data.shape) > 1 and data.shape[1] > 1:
                            print(f"Converting stereo to mono for {file}")
                            data = np.mean(data, axis=1).astype(data.dtype)
                        
                        train_data.append(data)
                        train_labels.append(idx)
                        train_files.append((model, file))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            else:
                print(f"No WAV files found in {model_train_dir}")
        else:
            print(f"Directory not found: {model_train_dir}")
            
        # Test data
        model_test_dir = os.path.join(TEST_DIR, model)
        print(f"Looking for test files in {model_test_dir}")
        
        if os.path.exists(model_test_dir):
            # Look for .wav or .WAV files
            model_test_files = [f for f in os.listdir(model_test_dir) 
                              if f.lower().endswith('.wav') and not f.startswith('.')]
                              
            if model_test_files:
                print(f"Found {len(model_test_files)} WAV files for {model} in test directory")
                
                for file in model_test_files:
                    try:
                        # Read the audio file
                        file_path = os.path.join(model_test_dir, file)
                        sample_rate, data = sci_wav.read(file_path)
                        
                        # Convert stereo to mono if needed
                        if len(data.shape) > 1 and data.shape[1] > 1:
                            print(f"Converting stereo to mono for {file}")
                            data = np.mean(data, axis=1).astype(data.dtype)
                        
                        test_data.append(data)
                        test_labels.append(idx)
                        test_files.append((model, file))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            else:
                print(f"No WAV files found in {model_test_dir}")
        else:
            print(f"Directory not found: {model_test_dir}")
    
    # Convert labels to numpy arrays
    train_labels = np.array(train_labels) if train_labels else np.array([])
    test_labels = np.array(test_labels) if test_labels else np.array([])
    
    # Print summary
    print(f"\nTraining set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Count samples per class
    if len(train_data) > 0:
        print("\nTraining samples per class:")
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  - {DRONE_MODELS[label]}: {count} samples")
    
    if len(test_data) > 0:
        print("\nTest samples per class:")
        unique_labels, counts = np.unique(test_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  - {DRONE_MODELS[label]}: {count} samples")
    
    return train_data, train_labels, test_data, test_labels, train_files, test_files


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


def show_confusion_matrix(title: str, confusion_matrix, class_names: list, save_path=None, normalize=True):
    """
    Generates plot of confusion matrix and optionally saves it to a file.

    :param title: Plot title
    :param confusion_matrix: Confusion matrix
    :param class_names: List of class names
    :param save_path: Optional path to save the plot as PNG
    :param normalize: Whether to normalize the confusion matrix (True) or show raw counts (False)
    """
    plt.figure(figsize=(14, 12))
    plt.title(title, fontsize=16)

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
    
    # Use the full folder names directly
    df_cm = pd.DataFrame(cm_display, index=class_names, columns=class_names)

    # Plot heatmap with improved layout for many classes
    cmap = "Blues" if normalize else "YlGnBu"
    heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt, cmap=cmap, 
                          square=True, linewidths=.5)

    # Adjust labels
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)
    
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
    plt.figure(figsize=(20, 10))
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
    train_data, train_labels, test_data, test_labels, train_files, test_files = read_data()
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Error: Insufficient data loaded. Please check your file paths.")
        print("\nPossible solutions:")
        print("1. Ensure your WAV files are in the expected locations")
        print("2. Make sure the filename or folder names contain the drone model identifiers")
        print("3. Modify the script to match your specific directory structure")
        sys.exit(1)

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
    print("\nExtracting MFCC features from training data...")
    X_train = preprocess_raw_data(train_data, fix_len, mfcc_parameters)
    X_train = np.array([feature.ravel() for feature in X_train])
    
    print("Extracting MFCC features from test data...")
    X_test = preprocess_raw_data(test_data, fix_len, mfcc_parameters)
    X_test = np.array([feature.ravel() for feature in X_test])

    # Use PCA for dimensionality reduction
    # Adjust n_components to be at most the number of samples - 1
    n_components = min(50, len(train_data) - 1)
    print(f"\nApplying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    # Print explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA explains {explained_variance:.2%} of variance")

    # Try both Logistic Regression and Random Forest
    print("\nTraining models...")
    
    # Count unique classes in training data
    n_classes = len(np.unique(train_labels))
    print(f"Training with {n_classes} unique drone classes")
    
    # 1. Logistic Regression with stronger regularization for many classes
    # logreg = LogisticRegression(C=1, solver='saga', multi_class='multinomial', max_iter=2000, class_weight='balanced')    # Replaced with line below.
    logreg = LogisticRegression(C=1, solver='saga', max_iter=2000, class_weight='balanced')

    logreg.fit(X_train, train_labels)
    
    # 2. Random Forest (often better for audio classification)
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, 
                               min_samples_split=5, random_state=42, 
                               class_weight='balanced')
    rf.fit(X_train, train_labels)
    
    # Predictions
    y_train_pred_lr = logreg.predict(X_train)
    y_test_pred_lr = logreg.predict(X_test)
    
    y_train_pred_rf = rf.predict(X_train)
    y_test_pred_rf = rf.predict(X_test)
    
    # Get actual model names being used (from the data we have)
    actual_model_names = [DRONE_MODELS[i] for i in sorted(np.unique(np.concatenate([train_labels, test_labels])))]
    
    # Print classification reports
    print("\n=== Logistic Regression Results ===")
    print('\nTrain classification report:')
    print(classification_report(train_labels, y_train_pred_lr, target_names=actual_model_names))
    
    print('\nTest classification report:')
    print(classification_report(test_labels, y_test_pred_lr, target_names=actual_model_names))
    
    print("\n=== Random Forest Results ===")
    print('\nTrain classification report:')
    print(classification_report(train_labels, y_train_pred_rf, target_names=actual_model_names))
    
    print('\nTest classification report:')
    print(classification_report(test_labels, y_test_pred_rf, target_names=actual_model_names))
    
    # Determine which model performed better on test data
    lr_accuracy = np.mean(y_test_pred_lr == test_labels)
    rf_accuracy = np.mean(y_test_pred_rf == test_labels)
    
    print(f"\nTest Accuracy: Logistic Regression: {lr_accuracy:.4f}, Random Forest: {rf_accuracy:.4f}")
    
    better_model = "Random Forest" if rf_accuracy > lr_accuracy else "Logistic Regression"
    better_preds = y_test_pred_rf if rf_accuracy > lr_accuracy else y_test_pred_lr
    
    print(f"Best model: {better_model}")
    
    # Print individual test file predictions for the better model
    print("\nIndividual test file predictions (using best model):")
    print("-" * 70)
    
    misclassified_count = 0
    
    for i, (true_label, prediction) in enumerate(zip(test_labels, better_preds)):
        true_model = DRONE_MODELS[true_label]
        predicted_model = DRONE_MODELS[prediction]
        model_name, file_name = test_files[i]
        
        # Check if prediction was correct and format output
        if true_label == prediction:
            result = "CORRECT"
        else:
            result = "WRONG"
            misclassified_count += 1
            
        print(f"{result}: {model_name}/{file_name} → Predicted: {predicted_model}")
    
    print(f"\nSummary: {misclassified_count} out of {len(test_labels)} files misclassified "
          f"({misclassified_count/len(test_labels)*100:.1f}%)")

    # Show confusion matrices
    print("\nGenerating confusion matrices...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create output directory for saving plots if it doesn't exist
    output_dir = os.path.join(ROOT_DIR, "model_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get the model labels
    train_model_labels = [DRONE_MODELS[i] for i in sorted(np.unique(train_labels))]
    test_model_labels = [DRONE_MODELS[i] for i in sorted(np.unique(test_labels))]
    
    if lr_accuracy > rf_accuracy:
        # Calculate confusion matrices once
        train_cm = confusion_matrix(train_labels, y_train_pred_lr)
        test_cm = confusion_matrix(test_labels, y_test_pred_lr)
        
        # Logistic Regression normalized matrices
        train_cm_path = os.path.join(output_dir, f"{timestamp}_lr_train_confusion_matrix_normalized.png")
        test_cm_path = os.path.join(output_dir, f"{timestamp}_lr_test_confusion_matrix_normalized.png")
        
        show_confusion_matrix("Logistic Regression - Train", 
                            train_cm, 
                            train_model_labels,
                            save_path=train_cm_path,
                            normalize=True)
                            
        show_confusion_matrix("Logistic Regression - Test", 
                            test_cm, 
                            test_model_labels,
                            save_path=test_cm_path,
                            normalize=True)
                            
        # Logistic Regression raw count matrices
        train_cm_counts_path = os.path.join(output_dir, f"{timestamp}_lr_train_confusion_matrix_counts.png")
        test_cm_counts_path = os.path.join(output_dir, f"{timestamp}_lr_test_confusion_matrix_counts.png")
        
        show_confusion_matrix("Logistic Regression - Train", 
                            train_cm, 
                            train_model_labels,
                            save_path=train_cm_counts_path,
                            normalize=False)
                            
        show_confusion_matrix("Logistic Regression - Test", 
                            test_cm, 
                            test_model_labels,
                            save_path=test_cm_counts_path,
                            normalize=False)
                            
        # Side-by-side comparison matrices
        train_comparison_path = os.path.join(output_dir, f"{timestamp}_lr_train_confusion_matrix_comparison.png")
        test_comparison_path = os.path.join(output_dir, f"{timestamp}_lr_test_confusion_matrix_comparison.png")
        
        show_confusion_matrix_comparison("Logistic Regression - Train", 
                                      train_cm, 
                                      train_model_labels,
                                      save_path=train_comparison_path)
                                      
        show_confusion_matrix_comparison("Logistic Regression - Test", 
                                      test_cm, 
                                      test_model_labels,
                                      save_path=test_comparison_path)
    else:
        # Calculate confusion matrices once
        train_cm = confusion_matrix(train_labels, y_train_pred_rf)
        test_cm = confusion_matrix(test_labels, y_test_pred_rf)
        
        # Random Forest normalized matrices
        train_cm_path = os.path.join(output_dir, f"{timestamp}_rf_train_confusion_matrix_normalized.png")
        test_cm_path = os.path.join(output_dir, f"{timestamp}_rf_test_confusion_matrix_normalized.png")
        
        show_confusion_matrix("Random Forest - Train", 
                            train_cm, 
                            train_model_labels,
                            save_path=train_cm_path,
                            normalize=True)
                            
        show_confusion_matrix("Random Forest - Test", 
                            test_cm, 
                            test_model_labels,
                            save_path=test_cm_path,
                            normalize=True)
                            
        # Random Forest raw count matrices
        train_cm_counts_path = os.path.join(output_dir, f"{timestamp}_rf_train_confusion_matrix_counts.png")
        test_cm_counts_path = os.path.join(output_dir, f"{timestamp}_rf_test_confusion_matrix_counts.png")
        
        show_confusion_matrix("Random Forest - Train", 
                            train_cm, 
                            train_model_labels,
                            save_path=train_cm_counts_path,
                            normalize=False)
                            
        show_confusion_matrix("Random Forest - Test", 
                            test_cm, 
                            test_model_labels,
                            save_path=test_cm_counts_path,
                            normalize=False)
        
        # Side-by-side comparison matrices
        train_comparison_path = os.path.join(output_dir, f"{timestamp}_rf_train_confusion_matrix_comparison.png")
        test_comparison_path = os.path.join(output_dir, f"{timestamp}_rf_test_confusion_matrix_comparison.png")
        
        show_confusion_matrix_comparison("Random Forest - Train", 
                                      train_cm, 
                                      train_model_labels,
                                      save_path=train_comparison_path)
                                      
        show_confusion_matrix_comparison("Random Forest - Test", 
                                      test_cm, 
                                      test_model_labels,
                                      save_path=test_comparison_path)
        
    print(f"\nConfusion matrix images saved to {output_dir} directory")
    
    # Print feature importance for Random Forest (helps understand what distinguishes drones)
    if rf_accuracy >= lr_accuracy:
        print("\nTop 10 feature importances from Random Forest:")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, n_components)):
            print(f"Feature {indices[i]}: {importances[indices[i]]:.6f}")
    
    plt.show()


    """
    
├── data/
│   └── train/
│       └── background/
│       └── drone/
│       └── helicopter/
│       └── X4_D1_MATRICE300/
│       └── X4_D2_MAVIC3/
│       └── X4_D3_MavicAir2S/
│       └── X4_D4_MavicAir2/
│       └── X4_D5_MavicMini2/
│       └── X4_D6_Mavic2Pro/
│       └── X4_D7_Mavic2Pro/
│       └── X4_D8_Mavic3/
│       └── X4_D9_Phantom4/
│       └── X4_D10_Mavic2Zoom/
│       └── X4_D11_MavicMini2/
│       └── X4_D17_Phantom4/
│       └── X6_D12_YuneecH520/
│       └── X6_D13_YuneecH520ERTK/
│       └── X6_D14_S900/
│       └── X6_D15_X6D/
│       └── X6_D16_Y6/
└── └── test/
│       └── background/
│       └── drone/
│       └── helicopter/
│       └── X4_D1_MATRICE300/
│       └── X4_D2_MAVIC3/
│       └── X4_D3_MavicAir2S/
│       └── X4_D4_MavicAir2/
│       └── X4_D5_MavicMini2/
│       └── X4_D6_Mavic2Pro/
│       └── X4_D7_Mavic2Pro/
│       └── X4_D8_Mavic3/
│       └── X4_D9_Phantom4/
│       └── X4_D10_Mavic2Zoom/
│       └── X4_D11_MavicMini2/
│       └── X4_D17_Phantom4/
│       └── X6_D12_YuneecH520/
│       └── X6_D13_YuneecH520ERTK/
│       └── X6_D14_S900/
│       └── X6_D15_X6D/
│       └── X6_D16_Y6/
    """