import random
import pandas as pd
import numpy as np
import inquirer
from sklearn.calibration import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import DistanceMetric
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.svm import SVC
from scipy.stats import mode
from scipy.spatial.distance import cdist

np.set_printoptions(suppress=True)

airlines_delay = pd.read_csv("airlines_delay.csv")
sample_size = int(len(airlines_delay) * 0.01)
random_indices = random.sample(range(len(airlines_delay)), sample_size)
airlines_delay = airlines_delay.iloc[random_indices]
mobile_data = pd.read_csv("train.csv")

le = LabelEncoder()
airlines_delay['Airline'] = le.fit_transform(airlines_delay['Airline'])
airlines_delay['AirportFrom'] = le.fit_transform(airlines_delay['AirportFrom'])
airlines_delay['AirportTo'] = le.fit_transform(airlines_delay['AirportTo'])
airlines_delay = airlines_delay.drop('Flight', axis=1)

X_airline = airlines_delay.iloc[:, :-1].values
Y_airline = airlines_delay.iloc[:, -1].values
X_mobile = mobile_data.iloc[:, :-1].values
Y_mobile = mobile_data.iloc[:, -1].values

X_train_airline, X_test_airline, y_train_airline, y_test_airline = train_test_split(X_airline, Y_airline, test_size=0.3, random_state=42)
X_train_mobile, X_test_mobile, y_train_mobile, y_test_mobile = train_test_split(X_mobile, Y_mobile, test_size=0.3, random_state=42)

dist_cont = DistanceMetric.get_metric('euclidean')
dist_disc = DistanceMetric.get_metric('hamming')

def custom_distance_train(x1, x2):
    x1_disc = x1[:, [1, 3, 5, 17, 18, 19]]
    x1_cont = x1[:, [i for i in range(x1.shape[1]) if i not in [1, 3, 5, 17, 18, 19]]]
    x2_disc = x2[:, [1, 3, 5, 17, 18, 19]]
    x2_cont = x2[:, [i for i in range(x2.shape[1]) if i not in [1, 3, 5, 17, 18, 19]]]


    dist_cont_mat = dist_cont.pairwise(x1_cont, x2_cont)
    dist_disc_mat = dist_disc.pairwise(x1_disc, x2_disc)

    combined_dist = dist_cont_mat + dist_disc_mat
    return combined_dist

def custom_distance_airlines(x1, x2):

    dist_cont_mat = dist_cont.pairwise(x1, x2)

    combined_dist = dist_cont_mat 
    return combined_dist

def knn(k, distance_fn, X_train, y_train, X_test):
    dist_mat = distance_fn(X_train, X_test)
    
    nn_indices = np.argpartition(dist_mat, k, axis=0)[:k, :]
    
    nn_labels = y_train[nn_indices]
    
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

def NearestNeighbors(k,distance_fun,X_train=None,y_train=None,X_test=None,y_test=None,X=None,Y=None,type="test_set"):
    knn_custom = KNN_Custom(k=k,distance_fn=distance_fun)
    
    if type == 'test_set':
        knn_custom.fit(X_train,y_train)
        y_pred = knn_custom.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
    elif type == 'cross_val':
        scoring = {'accuracy': 'accuracy',
                   'f1_weighted': 'f1_weighted'}
        cv_results = cross_validate(estimator=knn_custom, X=X, y=Y, cv=10, n_jobs=-1, scoring=scoring)
        f1 = cross_val_score(knn_custom, X, Y, cv=10, scoring='f1_macro')
        accuracy = cross_val_score(knn_custom, X, Y, cv=10, scoring='accuracy')
        accuracy = accuracy.mean()
        f1 = f1.mean()
        
    
    return accuracy,f1

def NaiveBayes(X_train=None,y_train=None,X_test=None,y_test=None,X=None,Y=None,file=None,type="test_set"):
    if file == 'train.csv':
        continuous_indices = [2,7]
        discrete_indices = [i for i in range(X.shape[1]) if i not in continuous_indices]

        continuous_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        discrete_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ])


        preprocessor = ColumnTransformer([    
            ('continuous', continuous_pipeline, continuous_indices),    
            ('discrete', discrete_pipeline, discrete_indices),
        ])

        preprocessor.fit(X)

        X_transformed = preprocessor.transform(X)

        continuous_data = X_transformed[:, :len(continuous_indices)]
        discrete_data = X_transformed[:, len(continuous_indices):]

        gaussian_nb = GaussianNB()

        multinomial_nb = MultinomialNB()

        gaussian_nb.fit(continuous_data, Y)
        multinomial_nb.fit(discrete_data, Y)
        if type == "test_set":
            accuracyG = accuracy_score(Y, gaussian_nb.predict(continuous_data))
            f1G = f1_score(Y, gaussian_nb.predict(continuous_data), average='weighted')

            accuracyM = accuracy_score(Y, multinomial_nb.predict(discrete_data))
            f1M = f1_score(Y, multinomial_nb.predict(discrete_data), average='weighted')

        elif type == "cross_val":
            scoring = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}
            gaussian_nb_scores = cross_validate(gaussian_nb, continuous_data, Y, cv=10, scoring=scoring)
            accuracyG = np.mean(gaussian_nb_scores['test_accuracy'])
            f1G = np.mean(gaussian_nb_scores['test_f1_weighted'])
            multinomial_nb_scores = cross_validate(multinomial_nb, discrete_data, Y, cv=10, scoring=scoring)
            accuracyM = np.mean(multinomial_nb_scores['test_accuracy'])
            f1M = np.mean(multinomial_nb_scores['test_f1_weighted'])

        return accuracyG,accuracyM,f1G,f1M
                
    elif file == "airlines_delay.csv":
        continuous_indices = []
        discrete_indices = [i for i in range(X.shape[1]) if i not in continuous_indices]

        continuous_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        discrete_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ])


        preprocessor = ColumnTransformer([    
            ('continuous', continuous_pipeline, continuous_indices),    
            ('discrete', discrete_pipeline, discrete_indices),
        ])

        # Fit the preprocessor to the data
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)#???????????????????????

        continuous_data = X_transformed[:, :len(continuous_indices)]
        discrete_data = X_transformed[:, len(continuous_indices):]

        multinomial_nb = MultinomialNB()
        multinomial_nb.fit(discrete_data, Y)
        if type == "test_set":
            accuracy = accuracy_score(Y, multinomial_nb.predict(discrete_data))
            f1 = f1_score(Y, multinomial_nb.predict(discrete_data), average='weighted')
        elif type == "cross_val":
            scoring = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}
            multinomial_nb_scores = cross_validate(multinomial_nb, discrete_data, Y, cv=10, scoring=scoring)
            accuracy = np.mean(multinomial_nb_scores['test_accuracy'])
            f1 = np.mean(multinomial_nb_scores['test_f1_weighted'])

        return accuracy,f1
            
def NeuralNetwork(hidden_layers, n_neurons1, n_neurons2=None,X_train=None,y_train=None,X_test=None,y_test=None,X=None,Y=None, activation_hidden=None,type="test_set"):
    if hidden_layers == 1:
        hidden_layer_sizes = (n_neurons1)
    elif hidden_layers == 2:
        if n_neurons2 is not None:
            hidden_layer_sizes = (n_neurons1, n_neurons2)
        else:
            raise ValueError("Invalid number of neurons for the second hidden layer.")
    else:
        raise ValueError("Invalid number of hidden layers. Must be 1 or 2.")
    
    
    nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation_hidden,
                        solver='sgd',
                        learning_rate='constant',
                        learning_rate_init=0.01,
                        max_iter=1000,
                        tol=0.001)
    
    nn.out_activation_ = 'softmax'

    if(type == "test_set"):
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        accuracy = nn.score(X_test, y_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
    elif type == "cross_val":
        accuracy = cross_val_score(nn, X, Y, cv=10)
        accuracy = accuracy.mean()
        y_pred = cross_val_predict(nn, X, Y, cv=10)
        f1 = f1_score(Y, y_pred, average='weighted')

    return accuracy,f1

def SVM(num_of_categories,C,gamma=None,X_train=None,y_train=None,X_test=None,y_test=None,X=None,Y=None,kernel=None,type="test_set"):
    if num_of_categories == 2:
        if kernel == 'rbf':
            model = SVC(kernel=kernel,C=C,gamma=gamma,decision_function_shape='ovo')
        elif kernel == 'linear':
            model = SVC(kernel=kernel,C=C,decision_function_shape='ovo')
    else:
        if kernel is not None:
            if kernel == 'rbf':
                model = SVC(kernel=kernel,C=C,gamma=gamma,decision_function_shape='ovr')
            elif kernel == 'linear':
                model = SVC(kernel=kernel,C=C,decision_function_shape='ovr')

    if type == "test_set":
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test,y_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
    elif type == "cross_val":
        scores = cross_val_score(model, X, Y, cv=10)
        accuracy = scores.mean()
        y_pred = cross_val_predict(model, X, Y, cv=10)
        f1 = f1_score(Y, y_pred, average='weighted')

    return accuracy,f1

def askNeuralNetworkQuestions(file):
    global X_train_airline
    global X_test_airline
    global y_test_airline
    global y_train_airline
    global X_airline
    global Y_airline
    global X_train_mobile
    global X_test_mobile
    global X_mobile
    global y_test_mobile
    global y_train_mobile
    global Y_mobile

    questions = [
        inquirer.Text('hidden_layers', message="Number of hidden layers (1 or 2):"),
        inquirer.List('activation_function', 
                       message="Select activation function:",
                       choices=['logistic', 'tanh', 'relu']),
        inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation']),
    ]
    
    answers = inquirer.prompt(questions)
    training_method = answers["training_method"]
    hidden_layers = int(answers['hidden_layers'])
    activation_function = answers["activation_function"]
    
    if hidden_layers == 1:
        n1_neurons = [50,100,200]
        if file == "airlines_delay.csv":
            if(training_method == "test_set"):
                for i in range(len(n1_neurons)):
                    print("(70-30 train/test)Running Neural Network with ",activation_function,"activation function and K1=",n1_neurons[i])
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],X_train=X_train_airline,y_train=y_train_airline,X_test=X_test_airline,y_test=y_test_airline,activation_hidden=activation_function)
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
            else:
                for i in range(len(n1_neurons)):
                    print("(10-fold cross vallidation)Running Neural Network with ",activation_function,"activation function and K1=",n1_neurons[i])
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],X=X_airline,Y=Y_airline,activation_hidden=activation_function,type="cross_val")
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
        else:
            if(training_method == "test_set"):
                for i in range(len(n1_neurons)):
                    print("(70-30 train/test)Running Neural Network with ",activation_function,"activation function and K1=",n1_neurons[i])
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],X_train=X_train_mobile,y_train=y_train_mobile,X_test=X_test_mobile,y_test=y_test_mobile,activation_hidden=activation_function)
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
            else:
                for i in range(len(n1_neurons)):
                    print("(70-30 train/test)Running Neural Network with ",activation_function,"activation function and K1=",n1_neurons[i])
                    mean_accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],X=X_mobile,Y=Y_mobile,activation_hidden=activation_function,type="cross_val")
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")

    elif hidden_layers == 2:
        n1_neurons = [50,100,200]
        n2_neurons = [25,50,100]
        if file == "airlines_delay.csv":
            if(training_method == "test_set"):
                for i in range(len(n1_neurons)):
                    print("(70-30 train/test)Running Neural Network with ",activation_function,"activation function and (K1,K2) = (",n1_neurons[i],n2_neurons[i],")")
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],n2_neurons[i],X_train_airline,y_train_airline,X_test_airline,y_test_airline,activation_hidden=activation_function)
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
            else:
                for i in range(len(n1_neurons)):
                    print("(10-fold cross vallidation)Running Neural Network with ",activation_function,"activation function and (K1,K2) = (",n1_neurons[i],n2_neurons[i],")")
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],n2_neurons[i],X=X_airline,Y=Y_airline,activation_hidden=activation_function,type="cross_val")
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
        else:
            if(training_method == "test_set"):
                for i in range(len(n1_neurons)):
                    print("(70-30 train/test)Running Neural Network with ",activation_function,"activation function and (K1,K2) = (",n1_neurons[i],n2_neurons[i],")")
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],n2_neurons[i],X_train_mobile,y_train_mobile,X_test_mobile,y_test_mobile,activation_hidden=activation_function)
                    print("Accuracy =",f1)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
            else:
                for i in range(len(n1_neurons)):
                    print("(10-fold cross vallidation)Running Neural Network with ",activation_function,"activation function and (K1,K2) = (",n1_neurons[i],n2_neurons[i],")")
                    accuracy,f1 = NeuralNetwork(hidden_layers,n1_neurons[i],n2_neurons[i],X=X_mobile,Y=Y_mobile,activation_hidden=activation_function,type="cross_val")
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
    else:
        print("Invalid number of hidden layers. Please enter 1 or 2.")
        return
    
    return answers
   
def askSVM(file):
    global X_train_airline
    global X_test_airline
    global y_test_airline
    global y_train_airline
    global X_airline
    global Y_airline
    global X_train_mobile
    global X_test_mobile
    global X_mobile
    global y_test_mobile
    global y_train_mobile
    global Y_mobile

    C_lst = [0.01,0.1,1,4,10]
    gamma_lst = [0.01,0.1,1,2,5,10]
    
    if file == 'airlines_delay.csv':
        questions = [
            inquirer.List('Kernel', 
                       message="Select Kernel:",
                       choices=['rbf', 'linear']),
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
        ]
        answers = inquirer.prompt(questions)
        if answers["Kernel"] == "rbf" and answers["training_method"] == "test_set":
            print("(70-30 test/train)Running SVC airlines_delay.csv. . .")
            for c in C_lst:
                for gamma in gamma_lst:
                    accuracy,f1 = SVM(4,c,gamma,X_train=X_train_airline,y_train=y_train_airline,X_test=X_test_airline,y_test=y_test_airline,kernel="rbf",type="test_set")
                    print("(70-30 TRAIN-TEST) C = ",c,"|gamma = ",gamma,"|Accuracy = ",accuracy,"F1-score = ",f1)
        elif answers["Kernel"] == "rbf" and answers["training_method"] == "cross_validation":
            print("(CROSS-VAL)Running SVC airlines_delay.csv. . .")
            for c in C_lst:
                for gamma in gamma_lst:
                    accuracy,f1 = SVM(4,c,gamma,X=X_airline,Y=Y_airline,kernel="rbf",type="cross_val")
                    print("(CROSS_VAL) C = ",c,"|gamma = ",gamma,"|Accuracy = ",accuracy,"F1-score = ",f1)
        elif answers["Kernel"] == "linear" and answers["training_method"] == "test_set":
            print("(70-30 TRAIN-TEST)Running SVC airlines_delay.csv. . .")
            for c in C_lst:
                accuracy,f1 = SVM(4,c,X_train=X_train_airline,y_train=y_train_airline,X_test=X_test_airline,y_test=y_test_airline,kernel="linear",type="test_set")
                print("(70-30 TRAIN-TEST) C = ",c,"|Accuracy = ",accuracy,"F1-score = ",f1)
        elif answers["Kernel"] == "linear" and answers["training_method"] == "cross_validation":
            print("(CROSS_VAL) Running SVC airlines_delay.csv. . .")
            for c in C_lst:
                accuracy,f1 = SVM(4,c,X=X_airline,Y=Y_airline,kernel="linear",type="cross_val")
                print("(CROSS_VAL) C = ",c,"|Accuracy = ",accuracy,"F1-score = ",f1)
    else:
        questions = [
            inquirer.List('kernel', 
                       message="Select kernel:",
                       choices=['rbf', 'linear'])
        ]
        answers = inquirer.prompt(questions)
        if answers["kernel"] == 'rbf':
            kernel = answers["kernel"]
            questions = [
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
            ]
            answers = inquirer.prompt(questions)
            
            if answers["training_method"] == "test_set":
                scaler = StandardScaler()
                X_train_mobile = scaler.fit_transform(X_train_mobile)
                X_test_mobile = scaler.transform(X_test_mobile)
                print("(70-30 test/train)Running SVC (kernel rbf). . .")
                for c in C_lst:
                    for gamma in gamma_lst:
                        accuracy,f1 = SVM(4,c,gamma,X_train=X_train_mobile,y_train=y_train_mobile,X_test=X_test_mobile,y_test=y_test_mobile,kernel=kernel,type="test_set")
                        print("Current C =",c)
                        print("Current gamma=",gamma)
                        print("Accuracy =",accuracy)
                        print("F1-Score =",f1)
                        print("-------------------------------------")
            else:
                print("(10-fold cross vallidation)Running SVC (kernel rbf). . .")
                scaler = StandardScaler()
                X_mobile = scaler.fit_transform(X_mobile)
                for c in C_lst:
                    for gamma in gamma_lst:
                        accuracy,f1 = SVM(4,c,gamma,X=X_mobile,Y=Y_mobile,kernel=kernel,type="cross_val")
                        print("Current C =",c)
                        print("Current gamma=",gamma)
                        print("Accuracy =",accuracy)
                        print("F1-Score =",f1)
                        print("-------------------------------------")
        else:
            kernel = answers["kernel"]
            questions = [
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
            ]
            answers = inquirer.prompt(questions)
            if answers["training_method"] == "test_set":
                scaler = StandardScaler()
                print("(70-30 test/train)Running SVC (kernel linear). . .")
                X_train_mobile = scaler.fit_transform(X_train_mobile)
                X_test_mobile = scaler.transform(X_test_mobile)
                for c in C_lst:
                    accuracy,f1 = SVM(4,c,X_train=X_train_mobile,y_train=y_train_mobile,X_test=X_test_mobile,y_test=y_test_mobile,kernel=kernel,type="test_set")
                    print("Current C =",c)
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")
            else:
                print("(10-fold cross vallidation)Running SVC (kernel linear). . .")
                scaler = StandardScaler()
                X_mobile = scaler.fit_transform(X_mobile)
                for c in C_lst:
                    accuracy,f1 = SVM(4,c,X=X_mobile,Y=Y_mobile,kernel=kernel,type="cross_val")
                    print("Current C =",c)
                    print("Accuracy =",accuracy)
                    print("F1-Score =",f1)
                    print("-------------------------------------")

def askKnnNearestNeighbors(file):
    global X_train_airline
    global X_test_airline
    global y_test_airline
    global y_train_airline
    global X_airline
    global Y_airline
    global X_train_mobile
    global X_test_mobile
    global X_mobile
    global y_test_mobile
    global y_train_mobile
    global Y_mobile

    k = [1,3,5,10]
    if file == 'airlines_delay.csv':
        print("Running Nearest Neighbors (airlines_delay). . .")
        questions = [
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
        ]
        answers = inquirer.prompt(questions)
        if answers["training_method"] == "test_set":
            scaler = StandardScaler()
            X_train_airline = scaler.fit_transform(X_train_airline)
            X_test_airline = scaler.transform(X_test_airline)
            for i in k:
                accuracy,f1 = NearestNeighbors(i,distance_fun=custom_distance_airlines,X_train=X_train_airline,y_train=y_train_airline,X_test=X_test_airline,y_test=y_test_airline,type="test_set")
                print("(70-30 TRAIN-TEST) Nearest Neighbors|k =",i,"|Accuracy=",accuracy,"|F1-score=",f1)
        elif answers["training_method"] == "cross_validation":
            scaler = StandardScaler()
            X_airline = scaler.fit_transform(X_airline)
            for i in k:
                accuracy,f1 = NearestNeighbors(i,distance_fun=custom_distance_airlines,X=X_airline,Y=Y_airline,type="cross_val")
                print("(CROSS VAL) Nearest Neighbors|k =",i,"|Accuracy=",accuracy,"|F1-score=",f1)
    elif file == 'train.csv':
        print("Running Nearest Neighbors (train.csv). . .")
        questions = [
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
        ]
        answers = inquirer.prompt(questions)
        if answers["training_method"] == "test_set":
            scaler = StandardScaler()
            X_train_mobile = scaler.fit_transform(X_train_mobile)
            X_test_mobile = scaler.transform(X_test_mobile)
            for i in k:
                accuracy,f1 = NearestNeighbors(i,distance_fun=custom_distance_train,X_train=X_train_mobile,y_train=y_train_mobile,X_test=X_test_mobile,y_test=y_test_mobile,type="test_set")
                print("(70-30 TRAIN-TEST) Nearest Neighbors|k =",i,"|Accuracy=",accuracy,"|F1-score=",f1)
        elif answers["training_method"] == "cross_validation":
            scaler = StandardScaler()
            X_mobile = scaler.fit_transform(X_mobile)
            for i in k:
                accuracy,f1 = NearestNeighbors(i,distance_fun=custom_distance_train,X=X_mobile,Y=Y_mobile,type="cross_val")
                print("(CROSS VAL) Nearest Neighbors|k =",i,"|Accuracy=",accuracy,"|F1-score=",f1)

def askNaiveBayes(file):
    global X_train_airline
    global X_test_airline
    global y_test_airline
    global y_train_airline
    global X_airline
    global Y_airline
    global X_train_mobile
    global X_test_mobile
    global X_mobile
    global y_test_mobile
    global y_train_mobile
    global Y_mobile

    questions = [
            inquirer.List('training_method', 
                       message="Select training method:",
                       choices=['test_set', 'cross_validation'])
        ]
    
    answers = inquirer.prompt(questions)
    scaler = StandardScaler()
    X_airline = scaler.fit_transform(X_airline)
    X_train_airline = scaler.fit_transform(X_train_airline)
    X_test_airline = scaler.transform(X_test_airline)
    X_mobile = scaler.fit_transform(X_mobile)
    X_train_mobile = scaler.fit_transform(X_train_mobile)
    X_test_mobile = scaler.transform(X_test_mobile)

    if file == "airlines_delay.csv":
        print("Running Naive Bayes (airlines_delay.csv). . .")
        if answers["training_method"] == "test_set":
            accuracy,f1 = NaiveBayes(X=X_airline,Y=Y_airline,file="airlines_delay.csv",type="test_set")
            print("(70-30 Test-Train) Accuracy = ",accuracy,"|F1-score =",f1)
        elif answers["training_method"] == "cross_validation":
            accuracy,f1 = NaiveBayes(X=X_airline,Y=Y_airline,file="airlines_delay.csv",type="cross_val")
            print("(CROSS-VAL) Accuracy = ",accuracy,"|F1-score =",f1)
    elif file == "train.csv":
        print("Running Naive Bayes (train.csv). . .")
        if answers["training_method"] == "test_set":
            accuracyG,accuracyM,f1G,f1M = NaiveBayes(X=X_mobile,Y=Y_mobile,file="train.csv",type="test_set")
            print("(70-30 TEST-TRAIN) Accuracy(Gaussian) = ",accuracyG,"|F1-score(Gaussian) =",f1G,"|Accuracy(Multinomial) = ",accuracyM,"|F1-score(Multinomial) =",f1M)
        elif answers["training_method"] == "cross_validation":
            accuracyG,accuracyM,f1G,f1M = NaiveBayes(X=X_mobile,Y=Y_mobile,file="train.csv",type="cross_val")
            print("(CROSS-VAL) Accuracy(Gaussian) = ",accuracyG,"|F1-score(Gaussian) =",f1G,"|Accuracy(Multinomial) = ",accuracyM,"|F1-score(Multinomial) =",f1M)

questions = [
  inquirer.List('File',
                message="Select csv file",
                choices=['airlines_delay.csv', 'train.csv'],
            ),
  inquirer.List('Methods',
                    message="Select Method",
                    choices=['K-NN Nearest Neighbors', 'Bayes', 'Neural Network', 'SVM'],
                    ),
]

answers = inquirer.prompt(questions)

if 'Neural Network' in answers["Methods"]:
    answers = askNeuralNetworkQuestions(answers["File"])
elif 'SVM' in answers["Methods"]:
    askSVM(answers["File"])
elif "K-NN Nearest Neighbors" in answers["Methods"]:
    askKnnNearestNeighbors(answers["File"])
elif "Bayes" in answers["Methods"]:
    askNaiveBayes(answers["File"])
