#---------------- K-NN Nearest Neighbors ----------------#
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import euclidean, cdist
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random

#---------------- 70 - 30 split ----------------#

dist_cont = DistanceMetric.get_metric('euclidean')
dist_disc = DistanceMetric.get_metric('hamming')

# set the random seed
random.seed(123)

# Load the data
np.set_printoptions(suppress=True)
#airlines_delay = pd.read_csv("D:\\Machine Learning\\1η Σειρά Ασκήσεων\\airlines_delay.csv", nrows=10000)
num_rows = 20000
skip_rows = sorted(random.sample(range(1, 1936759), 1936759 - num_rows))
airlines_delay = pd.read_csv("D:\\Machine Learning\\1η Σειρά Ασκήσεων\\airlines_delay.csv", skiprows=skip_rows)
le = LabelEncoder()
airlines_delay['Airline'] = le.fit_transform(airlines_delay['Airline'])
airlines_delay['AirportFrom'] = le.fit_transform(airlines_delay['AirportFrom'])
airlines_delay['AirportTo'] = le.fit_transform(airlines_delay['AirportTo'])
airlines_delay = airlines_delay.drop('Flight', axis=1)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(airlines_delay.iloc[:, :-1].values)
Y = airlines_delay.iloc[:, -1].values

def custom_distance(x1, x2):
    #x1_disc = x1[:, []]  # no discrete columns
    #x1_cont = x1[:, :]   # all columns are continuous
    #x2_disc = x2[:, []]  # no discrete columns
    #x2_cont = x2[:, :]   # all columns are continuous

    dist_cont_mat = dist_cont.pairwise(x1, x2)

    combined_dist = dist_cont_mat 
    return combined_dist


def knn(k, distance_fn, X_train, y_train, X_test):
    """
    k: number of neighbors to consider
    distance_fn: function that computes the distance between two points
    X_train: training feature matrix
    y_train: training target vector
    X_test: testing feature matrix
    """
    # compute distances between each pair of training and testing points
    dist_mat = distance_fn(X_train, X_test)
    
    # get the k-nearest neighbors for each testing point
    nn_indices = np.argpartition(dist_mat, k, axis=0)[:k, :]
    
    # get the labels for the nearest neighbors
    nn_labels = y_train[nn_indices]
    
    # predict the class for each testing point based on the majority class of its neighbors
    y_pred = np.empty(X_test.shape[0])
    for i in range(X_test.shape[0]):
        counter = Counter(nn_labels[:,i])
        y_pred[i] = counter.most_common(1)[0][0]
    
    return y_pred

class KNN_Custom:
    def __init__(self, k=5, distance_fn='euclidean'):
        self.k = k
        self.distance_fn = distance_fn
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        return knn(self.k, self.distance_fn, self.X_train, self.y_train, X)
    
    def score(self, X, y, scoring='f1'):
        y_pred = self.predict(X)
        if scoring == 'f1':
            return f1_score(y, y_pred, average='weighted')
        elif scoring == 'accuracy':
            return accuracy_score(y, y_pred)
        else:
            raise ValueError("Invalid scoring metric")

    def get_params(self, deep=True):
        return {'k': self.k, 'distance_fn': self.distance_fn}

class KNN_Euclidean:
    def __init__(self, k=5, distance_fn='euclidean'):
        self.k = k
        self.distance_fn = distance_fn
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        distances = cdist(X, self.X_train, metric=self.distance_fn)
        indices = np.argsort(distances, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[indices]
        y_pred = mode(k_nearest_labels, axis=1)[0].ravel()
        return y_pred
    
    def score(self, X, y, scoring='f1'):
        y_pred = self.predict(X)
        if scoring == 'f1':
            return f1_score(y, y_pred, average='weighted')
        elif scoring == 'accuracy':
            return accuracy_score(y, y_pred)
        else:
            raise ValueError("Invalid scoring metric")

    def get_params(self, deep=True):
        return {'k': self.k, 'distance_fn': self.distance_fn}

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# create an instance of the KNN class with custom distance function
knn_custom = KNN_Custom(k=10, distance_fn=custom_distance)
# create an instance of the KNN class with euclidean distance 
#knn_custom = KNN_Euclidean(k=10, distance_fn=euclidean)
#knn_custom = KNeighborsClassifier(10)

knn_custom.fit(X_train, y_train)

# Define the metrics to be evaluated
scoring = {'accuracy': 'accuracy',
           'f1_weighted': 'f1_weighted'}

# Fit the model to the training data
knn_custom.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn_custom.predict(X_test)

# Evaluate model performance on the testing data
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print('Accuracy score:', accuracy)
print('F1 score (weighted):', f1)

