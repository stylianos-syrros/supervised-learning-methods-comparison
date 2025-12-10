# ML Model Comparison: kNN, Neural Networks, SVMs & Naive Bayes

This folder contains a small machine learning project where I experimented with
several classic classification algorithms on different datasets.
The goal was to understand how each model behaves in practice, how sensitive it is
to hyperparameters, and how evaluation metrics change across datasets.

## Project Overview

The code focuses on comparing four families of models:

1. **k-Nearest Neighbors (kNN)**
2. **Feed-forward Neural Networks (MLP)**
3. **Support Vector Machines (SVMs)**
4. **Naive Bayes**

The experiments are implemented in Python using:
- `scikit-learn` for kNN, SVMs and Naive Bayes
- `TensorFlow / Keras` for the neural networks
- `pandas` / `numpy` for data handling

The work originally started in the context of a machine learning course, but the
code here is organized and documented as a standalone project.

## Datasets

The project uses tabular datasets related to flight delays and another
classification problem.

- **Airline delay dataset (`airlines_delay.csv`)**
  - Flight-level information (airline, origin/destination airports, delay label, etc.)
  - Categorical features are encoded using `LabelEncoder`
  - A random 1% subsample is often used for faster experimentation

- **Mobile / tabular dataset (`train.csv`)**
  - A second tabular classification problem
  - Used to see how the same models behave on a different feature space

> Note: `airlines_delay.csv` is relatively large. In some setups it might be
> preferable to keep it outside version control and download it separately.

## Methods

### 1. k-Nearest Neighbors (kNN)

- Implemented with scikit-learn’s `KNeighborsClassifier`
- Experiments with different values of **K** (e.g. 1, 5, 10)
- Multiple distance metrics:
  - **Euclidean** distance on continuous features
  - **Cosine** / **Hamming** distance on discrete features
  - A custom distance that combines continuous and discrete parts

For the image-style experiments (when used), each input example is represented as
high-dimensional feature vectors (e.g. flattened images).

Typical observations:
- Performance is quite stable across different K values
- Accuracy and F1 score often peak around intermediate values of K

### 2. Neural Networks

- Implemented with `TensorFlow` / `Keras`
- Fully-connected feed-forward networks (MLPs) with:
  - 1 or 2 hidden layers
  - Hidden layers with hundreds of neurons
  - Activation functions such as **sigmoid** and **ReLU**
  - `softmax` output layer for multi-class classification

Example experiments:
- Compare architectures such as:
  - 1 hidden layer with 500 neurons
  - 2 hidden layers with 500 and 200 neurons
- Measure accuracy on the test set and see how depth and activation functions
  affect performance

### 3. Support Vector Machines (SVMs)

- Implemented with scikit-learn’s `SVC`
- Multi-class classification is handled with a **One-vs-Rest (OvR)** strategy
- Different kernel functions are tested:
  - **Linear**
  - **RBF / Gaussian**
  - **Cosine** (via a custom kernel or precomputed similarity)

In practice, the Gaussian (RBF) kernel often achieves the best trade-off between
accuracy and F1 score on these datasets.

### 4. Naive Bayes

- Implemented as a **Gaussian Naive Bayes** classifier
- Assumes that features are conditionally independent given the class
- Each feature is modeled with a Gaussian distribution per class

Naive Bayes is very fast to train and works surprisingly well in some settings,
although the independence assumption is quite strong.

## Evaluation

The models are evaluated using:

- **Accuracy**
- **F1 score**
- Train / test splits (e.g. 70% / 30%)
- In some scripts: k-fold cross-validation on tabular datasets

Several scripts in this folder:
- Train models on the airline delay and mobile datasets
- Compare simple train/test splits versus cross-validation
- Print or log the resulting metrics for each configuration

## File overview

- **KNN.py**
  - Trains a k-Nearest Neighbors classifier on `train.csv` and produces predictions
    for `test.csv`. Encodes the `color` feature numerically and saves the results
    as `KNN_output.csv`.

- **NeuralNetwork.py**
  - Builds a feed-forward neural network (MLP) using `sklearn.neural_network.MLPClassifier`.
    Lets you choose the number of hidden layers and neurons from the console and writes
    predictions for `test.csv` to `NeuralNetwork_output.csv`.

- **SVM.py**
  - Trains a Support Vector Machine classifier (`SVC`) on `train.csv`. Allows choosing
    between a linear and an RBF (Gaussian) kernel and optionally setting the `gamma`
    parameter. Saves predictions for `test.csv` to `SVM_output.csv`.

- **NaiveBayes.py**
  - Uses a Gaussian Naive Bayes classifier on `train.csv` and writes the predicted
    labels for `test.csv` to `NaiveBayes_output.csv`.

- **Set1_4805.py** (and similar experiment scripts)
  - Runs experiments on the airline delay and mobile datasets. Handles data loading,
    encoding of categorical variables, train/test splits and custom distance metrics
    (e.g. combining Euclidean and Hamming) that are used by some of the models.

- **Set_Airlines_delay_*.py**
  - Focused experiments on the airline delay dataset using different algorithms
    (e.g. kNN, Naive Bayes). There are versions that use a simple train/test split
    and versions that perform k-fold cross-validation.

- **Set_Train_*.py**
  - Similar to the above, but targeting the dataset in `train.csv`. Used to compare
    how the same algorithms behave on a different feature space.

- **Test.py / Test2.py**
  - Utility scripts used to try out variations of the models, parameters or data
    preprocessing steps before integrating ideas into the main experiment files.

## How to Run

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Example: run one of the experiments
python KNN.py
python NeuralNetwork.py
python SVM.py
python NaiveBayes.py
```

## Dependencies

Main Python packages:

- numpy
- pandas
- scikit-learn
- tensorflow (or tensorflow-cpu)
- matplotlib

A minimal `requirements.txt` could look like:

```text
numpy
pandas
scikit-learn
tensorflow
matplotlib
```

This project is mainly about hands-on experimentation with classic ML models:
understanding distance metrics, tuning hyperparameters, and comparing how
different algorithms perform on the same datasets.
