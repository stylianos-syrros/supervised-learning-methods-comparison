# ML Model Comparison: kNN, Neural Networks, SVMs & Naive Bayes

This folder contains a machine learning project where I experimented with
several classic supervised classification algorithms across multiple datasets.
The goal was to understand how each model behaves in practice, how sensitive it is
to hyperparameters, and how evaluation metrics change across datasets.

> A hands-on comparison of classic supervised learning algorithms  
> *(kNN, SVMs, Naive Bayes, MLP Neural Networks)*  
> on multiple tabular classification datasets.  
> Includes custom distance metrics, cross-validation experiments,  
> and detailed evaluations.

---

## 📊 Sample Results (on the `train.csv` classification dataset)

Below is a small snapshot of representative model performance:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **kNN (K=5, cosine distance)** | **0.858** | **0.856** |
| **SVM (RBF kernel)** | **0.864** | **0.863** |
| **Neural Network (ReLU MLP)** | **0.893** | — |
| **Naive Bayes** | **0.625** | **0.630** |

*Note: These values come from experiments on the `train.csv` dataset and are included only as indicative results to illustrate typical performance differences between the models.*

---

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

The work originally started in the context of a machine learning course,
but the repository is structured and documented as an independent project.

---

## Datasets

Two main tabular datasets were used:

### **Airline delay dataset (`airlines_delay.csv`)**
- Contains flight-level information (airline, airports, delay labels, etc.)
- Categorical features are encoded using `LabelEncoder`
- Often a 1% subsample is used for faster experimentation

### **Mobile/tabular dataset (`train.csv`)**
- A second classification dataset
- Useful to compare model behavior across different feature spaces

> Note: `airlines_delay.csv` is large, so it may be preferable  
> to keep it outside version control and load it locally.

---

## Methods

### 1. k-Nearest Neighbors (kNN)

- Implemented with `KNeighborsClassifier`
- Tested values of K = 1, 5, 10
- Distance metrics:
  - Euclidean for continuous variables
  - Cosine / Hamming for categorical variables
  - A **hybrid custom distance** mixing Euclidean + Hamming

### 2. Neural Networks (MLP)

- Fully connected feed-forward networks using TensorFlow/Keras
- Architectures tested:
  - 1×500 hidden layer
  - 500 → 200 two-layer networks
- Activation functions: **sigmoid**, **ReLU**
- Output layer: softmax

### 3. Support Vector Machines (SVMs)

- Implemented with `SVC`
- One-vs-Rest strategy for multi-class cases
- Kernels tested:
  - Linear
  - RBF (Gaussian)
  - Cosine (via custom kernel)

### 4. Naive Bayes

- Gaussian Naive Bayes
- Extremely fast training time
- Performs surprisingly well on some tabular settings

---

## Evaluation

The models are evaluated using:

- **Accuracy**
- **F1 score**
- **Train/test splits** (70/30)
- **k-fold cross-validation** for some experiments

Scripts in the repository allow testing each model under multiple conditions
(train/test split vs cross-validation, different feature encodings, etc.)

---

## File overview

- **KNN.py**
  - kNN classifier on `train.csv`, writing predictions to `KNN_output.csv`.

- **NeuralNetwork.py**
  - MLP classifier with user-defined hidden layers.
  - Writes predictions to `NeuralNetwork_output.csv`.

- **SVM.py**
  - Linear / Gaussian SVM with optional `gamma` parameter.
  - Outputs predictions to `SVM_output.csv`.

- **NaiveBayes.py**
  - Gaussian Naive Bayes applied to `train.csv`.
  - Outputs predictions to `NaiveBayes_output.csv`.

- **Set1_4805.py**
  - Main experimental framework: dataset loading, encoding, train/test splits,
    and combined Euclidean + Hamming distance metrics.

- **Set_Airlines_delay_*.py**
  - Airline-delay-specific experiments using kNN or Naive Bayes  
    (with versions for train/test split and cross-validation).

- **Set_Train_*.py**
  - Equivalent experiments for the second dataset (`train.csv`).

- **Test.py / Test2.py**
  - Small utility scripts for testing parameters, preprocessing variations, etc.

---

## 📁 Repository Structure

```text
├── KNN.py
├── NeuralNetwork.py
├── SVM.py
├── NaiveBayes.py
│
├── Set1_4805.py
│
├── Set_Airlines_delay_K_Nearest_Neighbors_Split.py
├── Set_Airlines_delay_K_Nearest_Neighbors_Cross_Validation.py
├── Set_Airlines_delay_Naive_Bayes_Split.py
├── Set_Airlines_delay_Naive_Bayes_Cross_Validation.py
│
├── Set_Train_K_Nearest_Neighbors_Split.py
├── Set_Train_K_Nearest_Neighbors_Cross_Validation.py
├── Set_Train_Naive_Bayes_Split.py
├── Set_Train_Naive_Bayes_Cross_Validation.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Run any model
python KNN.py
python NeuralNetwork.py
python SVM.py
python NaiveBayes.py
```

🎛️ Running the Main Interactive Program (Set1_4805.py)
Although each model script (e.g., KNN.py, SVM.py) can run independently,
the main way to use the project is through the interactive controller:
👉 Set1_4805.py
This script provides a full menu-driven interface that lets you:
- Select which dataset to use
- Choose a machine learning method
- Choose an evaluation method
- Configure hyperparameters (when needed)

▶ Start the interface:

```bash
python Set1_4805.py
```

🧭 Interactive Menu Navigation

The program uses the `inquirer` library to display interactive terminal menus:
- Navigate using arrow keys (↑ ↓)
- Press Enter to select
- Prompts update dynamically based on your previous choices

📌 Menu Flow Overview

1️⃣ **Select CSV File**

Choose between:
- `airlines_delay.csv` → flight delay classification
- `train.csv` → mobile/tabular classification

The script automatically performs preprocessing (label encoding + scaling).

2️⃣ **Select Method**

You choose which model to run:
- k-NN Nearest Neighbors
- Bayes (Naive Bayes)
- Neural Network (MLP)
- SVM

Each method opens additional menus for model parameters.

3️⃣ **Select Training Method**

- `test_set` → 70/30 Train–Test split
- `cross_validation` → 10-Fold Cross Validation

⚙️ Hyperparameter Prompts

Depending on the selected method:

🔹 **kNN**
- Select value of K (e.g., 1, 3, 5, 10)

🔹 **SVM**
- Select kernel (linear or RBF)
- Select C value
- If RBF → select gamma value

🔹 **Neural Network**
- Select number of hidden layers (1 or 2)
- Select neurons per layer
- Select activation function (logistic or tanh)

🔹 **Naive Bayes**
- Runs immediately (no parameters).

📤 Output

Each experiment prints:
- Accuracy
- F1-score

For cross-validation experiments:
- Mean Accuracy
- Mean F1-score

This makes `Set1_4805.py` the central hub for running all comparisons in the project.

## Dependencies

Minimal Python requirements:

- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib
- inquirer

## 🎯 Goal of the Project

This project is about hands-on experimentation with classic supervised learning models:
understanding how distance metrics influence kNN, how hyperparameters affect neural networks,
and how different algorithms compare when applied to the same datasets.

It serves as a practical exploration of model behavior, evaluation methods, and
feature engineering choices in real-world tabular machine learning.
