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
gaussian_nb.fit(X_transformed, Y)
multinomial_nb.fit(X_transformed, Y)

# Define the metrics to be evaluated
scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}

# Use cross-validation to evaluate the performance of the classifiers
gaussian_nb_scores = cross_validate(gaussian_nb, X_transformed, Y, cv=10, scoring=scoring)
multinomial_nb_scores = cross_validate(multinomial_nb, X_transformed, Y, cv=10, scoring=scoring)

print("GaussianNB Scores:")
print("Accuracy:", np.mean(gaussian_nb_scores['test_accuracy']))
print("F1 macro:", np.mean(gaussian_nb_scores['test_f1_macro']))

print("\nMultinomialNB Scores:")
print("Accuracy:", np.mean(multinomial_nb_scores['test_accuracy']))
print("F1 macro:", np.mean(multinomial_nb_scores['test_f1_macro']))
'''   
# Define the pipelines for continuous and discrete features
continuous_pipeline = make_pipeline(
    StandardScaler(),
    GaussianNB()
)

discrete_pipeline = make_pipeline(
    MultinomialNB()
)

# Combine the pipelines using a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_pipeline, continuous_indices),
        ('discrete', discrete_pipeline, discrete_indices)
    ]
)

# Create the composite pipeline by combining the preprocessor with the classifier
nb_classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('estimator', NaiveBayesEstimator(continuous_indices, discrete_indices))
])

# Define the metrics to be evaluated
scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}

# Evaluate the classifier using 10-fold cross-validation
cv_results = cross_validate(nb_classifier, X, Y, cv=10, scoring=scoring)

# Print the cross-validation results
print("Accuracy scores:", cv_results['test_accuracy'])
print("F1 macro scores:", cv_results['test_f1_macro'])
print("Mean accuracy:", np.mean(cv_results['test_accuracy']))
print("Mean F1 macro:", np.mean(cv_results['test_f1_macro']))
'''

'''
# Perform 10-fold cross-validation and print the scores
scoring = {'accuracy': make_scorer(accuracy_score), 'f1_macro': make_scorer(f1_score, average='macro')}
cv_scores = cross_validate(nb, X, Y, cv=10, scoring=scoring)
print("Accuracy scores:", cv_scores['test_accuracy'])
print("F1 macro scores:", cv_scores['test_f1_macro'])
print("Mean accuracy:", np.mean(cv_scores['test_accuracy']))
print("Mean F1 macro:", np.mean(cv_scores['test_f1_macro']))'''

