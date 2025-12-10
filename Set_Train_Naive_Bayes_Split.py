#---------------- Naive Bayes Classifier ----------------#
from scipy.stats import norm, multinomial
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import euclidean
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.mixture import BayesianGaussianMixture
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, KBinsDiscretizer, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

#---------------- 70 - 30 split ----------------#
# Load the data
train_mobile = pd.read_csv("D:\\Machine Learning\\1η Σειρά Ασκήσεων\\train.csv")

X = train_mobile.iloc[:, :-1].values
Y = train_mobile.iloc[:, -1].values

# Define the indices of categorical and continuous features
continuous_indices = [2,7] # replace with the indices of your categorical features
discrete_indices = [i for i in range(X.shape[1]) if i not in continuous_indices]

# Create preprocessing pipelines for continuous and discrete features
continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

discrete_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder())
])


# Combine the preprocessing pipelines using ColumnTransformer
preprocessor = ColumnTransformer([    
    ('continuous', continuous_pipeline, continuous_indices),    
    ('discrete', discrete_pipeline, discrete_indices),
])

# Fit the preprocessor to the data
preprocessor.fit(X)

# Transform the data using the preprocessor
X_transformed = preprocessor.transform(X)

# Separate the continuous and discrete features
continuous_data = X_transformed[:, :len(continuous_indices)]
discrete_data = X_transformed[:, len(continuous_indices):]

# Create an instance of GaussianNB classifier for the continuous data
gaussian_nb = GaussianNB()

# Create an instance of MultinomialNB classifier for the discrete data
multinomial_nb = MultinomialNB()

# Fit the classifiers to the data
gaussian_nb.fit(continuous_data, Y)
multinomial_nb.fit(discrete_data, Y)

# Define the metrics to be evaluated
scoring = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Calculate the accuracy and f1 score of each classifier on the entire dataset
gaussian_nb_accuracy = accuracy_score(Y, gaussian_nb.predict(continuous_data))
gaussian_nb_f1_score = f1_score(Y, gaussian_nb.predict(continuous_data), average='weighted')

multinomial_nb_accuracy = accuracy_score(Y, multinomial_nb.predict(discrete_data))
multinomial_nb_f1_score = f1_score(Y, multinomial_nb.predict(discrete_data), average='weighted')

# Print the accuracy and f1 score of each classifier
print("GaussianNB Scores:")
print('Accuracy:', gaussian_nb_accuracy)
print('F1 weighted:', gaussian_nb_f1_score)

print("\nMultinomialNB Scores:")
print("Accuracy:", multinomial_nb_accuracy)
print("F1 weighted:", multinomial_nb_f1_score)

print("\nScores for whole X dataset:")
print("Accuracy:", 0.9*(multinomial_nb_accuracy) + 0.1*(gaussian_nb_accuracy))
print("F1 weighted:", 0.9*(multinomial_nb_f1_score) + 0.1*(gaussian_nb_f1_score))
