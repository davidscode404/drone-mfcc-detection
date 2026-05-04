# drone-mfcc-detection

<p align="justify">
Drone audio classification using MFCC (mel-frequency cepstral coefficients). There are two scripts, <code>drone_classification.py</code> classifies audio as drone, helicopter, or background noise, and <code>model_classification.py</code> identifies the specific drone model across 17 classes. Both use PCA (principal component analysis, a technique for reducing the number of features while retaining the most useful information) for dimensionality reduction; <code>drone_classification.py</code> uses logistic regression, while <code>model_classification.py</code> compares logistic regression and random forest, selecting the more accurate of the two. <code>drone_classification.py</code> is trained on a dataset of 90 recordings, 30 in each category; half were used for training, and half for testing. <code>model_classification.py</code> is trained on a dataset of 1513 recordings, 748 (44 for each of the 17 models) for training and 765 (45 for each of the 17 models) for testing.
</p>

Drone audio classification using MFCC (mel-frequency cepstral coefficients). There are two scripts, `drone_classification.py` classifies audio as drone, helicopter, or background noise, and `model_classification.py` identifies the specific drone model across 17 classes. Both use PCA (principal component analysis, a technique for reducing the number of features while retaining the most useful information) for dimensionality reduction; `drone_classification.py` uses logistic regression, while `model_classification.py` compares logistic regression and random forest, selecting the more accurate of the two. `drone_classification.py` is trained on a dataset of 90 recordings, 30 in each category; half were used for training, and half for testing. `model_classification.py` is trained on a dataset of 1513 recordings, 748 (44 for each of the 17 models) for training and 765 (45 for each of the 17 models) for testing.

## Results

`drone_classification.py` achieves 66.7% accuracy:

![](drone_results/20260504_222017_test_confusion_matrix_comparison.png)

`model_classification.py` achieves 51.4% accuracy:

![](model_results/20260504_225151_rf_test_confusion_matrix_comparison.png)

## Setup

Be sure to use a virtual environment:
> ```sh
> $ python3 -m venv venv
> ```
(Here, `venv` is just an example, the virtual environment can be given *any* name).

Activate the virtual environment:
> ```sh
> $ source venv/bin/activate
> ```

Upon activation, run `pip list`. Only `pip` and `setuptools` should be installed in the virtual environment.

Install requirements:
> ```sh
> $ pip install -r requirements.txt
> ```

## Classify drones:

`cd` into the `main` directory:
> ```sh
> $ cd main
> ```

Run `drone_classification.py`:
> ```sh
> $ python3 drone_classification.py
> ```

Six `png` filess will be saved to the drone_results folder:
- `_train_confusion_matrix_normalized.png` — single heatmap, training data, values as proportions (0.0–1.0).
- `_train_confusion_matrix_counts.png` — single heatmap, training data, values as raw file counts.
- `_train_confusion_matrix_comparison.png` — side-by-side of the above two, training data.
- `_test_confusion_matrix_normalized.png` — single heatmap, test data, values as proportions.
- `_test_confusion_matrix_counts.png` — single heatmap, test data, values as raw file counts.
- `_test_confusion_matrix_comparison.png` — side-by-side of the above two, test data.

Note that each `png` is prefaced with a timestamp (e.g. `20260504_143022_`).

## Classify drone models:

`cd` into the `main` directory:
> ```sh
> $ cd main
> ```

Run `model_classification.py`:
> ```sh
> $ python3 model_classification.py
> ```

Six `png` filess will be saved to the model_results folder:
- `_{lr/rf}_train_confusion_matrix_normalized.png` — training data, proportions
- `_{lr/rf}_train_confusion_matrix_counts.png` — training data, raw counts
- `_{lr/rf}_train_confusion_matrix_comparison.png` — training data, side-by-side
- `_{lr/rf}_test_confusion_matrix_normalized.png` — test data, proportions
- `_{lr/rf}_test_confusion_matrix_counts.png` — test data, raw counts
- `_{lr/rf}_test_confusion_matrix_comparison.png` — test data, side-by-side

Note that each `png` is prefaced with a timestamp (e.g. `20260504_143022_`), and either `lr_` (logistic regression) or `rf_` (random forest), depending on which classifier won on test accuracy.

## Dataset

The drone audio recordings used in `model_classification.py` are sourced from the following publication:

Mięsikowska, M. (2024). Classification of Unmanned Aerial Vehicles Based on Acoustic Signals Obtained in External Environmental Conditions. *Sensors*, 24(17), 5663. https://doi.org/10.3390/s24175663

Dataset available at: https://t47-marzena.s3.kielce.pl/index.html (CC BY 4.0)