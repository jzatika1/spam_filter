# Spam Filter Project

## Project Structure

spam_filter/
├── data/ # Directory for training data files
├── input/ # Directory for input text files for prediction
├── logs/ # Directory for log files
├── models/ # Directory for saved models
├── output/ # Directory for output prediction files
├── train_model.py # Script for training the Naive Bayes model
├── predict.py # Script for making predictions using the trained model
├── utils/ # Utility modules
│   ├── __init__.py
│   ├── data_loader.py # Module for loading and preprocessing data
│   ├── evaluation.py # Module for model evaluation
│   ├── logger.py # Module for setting up logging
├── README.md # Project documentation
├── requirements.txt # List of project dependencies

## Conda Environment Setup

To create a new conda environment with Python 3.12 and the required packages, use the following steps:

```shell
conda create -n spam_filter_env python=3.12
```

Once the environment is created, activate it:

```shell
conda activate spam_filter_env
```

Confirm Python version is 3.12:

```shell
python --version
```

Then, install the required packages:

```shell
python -m pip install -r requirements.txt
```

You can then run the training and prediction scripts.

## Scripts

### `train_model.py`

This script is used for training the Naive Bayes model. It supports hyperparameter optimization.

#### Usage

```shell
python train_model.py --model naive_bayes --training_data_directory data/ --training_data_files "*"
```

- `--model`: Specify the model to use (e.g., `naive_bayes`).
- `--training_data_directory`: Path to the folder containing training data files.
- `--training_data_files`: Names of the training data files (use quotes and * for all files in directory).
- `--hyperparameter_optimization`: (Optional) Perform hyperparameter optimization.

### `predict.py`

This script is used for making predictions using the trained model.

#### Usage

```shell
python predict.py --input_directory input/ --input_files "*"
```

- `--input_directory`: Path to the input directory containing text files.
- `--input_files`: Names of the input text files (use quotes and * for all files in directory).
- `--model_folder`: Path to the folder containing the trained model.
- `--output_folder`: Path to the folder where the output will be saved.