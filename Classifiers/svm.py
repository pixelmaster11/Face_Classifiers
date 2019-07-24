

import numpy as np
from sklearn.svm import LinearSVC, SVC
from Classifiers import ml_utils
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import utilities
from Classifiers.classifier import MLClassifier

class SVMClassifier():

    def __init__(self, model_name = "SVC"):

        self.name = model_name

        if model_name == "SVC":
            self.model = SVC(gamma="scale")#verbose=0, C=1, random_state=11, gamma=1, kernel="rbf", degree=2, probability=True)


        elif model_name == "LinearSVC":
            self.model = LinearSVC(random_state=22, C=1)

    def find_best_model_random(self, feat_train, lab_train):

        print("\nFinding best parameters using random search..")

        # Scale the input features
        feat_train = ml_utils.get_scaing(scaling_type="Norm").fit(feat_train).transform(feat_train)

        print("Features after Scaling: {}".format(np.array(feat_train).shape))

        # C
        C = [.001, .01, 1, 10, 100]

        # gamma
        gamma = [.0001, .001, .01, .1, 1, 10, 100]

        # degree
        degree = [1, 2, 3, 4, 5, 6, 7, 8, 9 , 10]

        # kernel
        kernel = ['linear', 'rbf', 'poly']

        # probability
        probability = [True]

        # Create the random grid
        random_grid = {'C': C,
                       'kernel': kernel,
                       'gamma': gamma,
                       'degree': degree,
                       'probability': probability
                       }

        # Definition of the random search
        random_search = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=random_grid,
                                           n_iter=50,
                                           scoring='accuracy',
                                           cv=5,
                                           verbose=1,
                                           random_state=22, n_jobs=-1)

        random_search.fit(feat_train, lab_train)

        print("\nThe best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(random_search.best_score_)

        self.find_best_model_grid(feat_train=feat_train, lab_train=lab_train)

        self.model = random_search.best_estimator_

    def find_best_model_grid(self, feat_train, lab_train):

        print("\nFinding best model using grid search..")

        # Scale the input features
        feat_train = ml_utils.get_scaing(scaling_type="Norm").fit(feat_train).transform(feat_train)

        print("Features after Scaling: {}".format(np.array(feat_train).shape))

        # Create the parameter grid based on the results of random search
        C = [0.1, 1, 2, 5, 10, 100]
        degree = [7, 8, 9]
        gamma = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1]
        probability = [True]


        param_grid = [
            {'C': C, 'kernel': ['linear'], 'probability': probability},
            {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
            {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
        ]


        cv = ml_utils.get_spliting(n_splits=3, test_size=0.33, random_state=22)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv,
                                   verbose=1,
                                   n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(feat_train, lab_train)

        print("\nThe best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)
        self.model = grid_search.best_estimator_

if __name__ == '__main__':

    features, labels, ips = utilities.load_embeddings(load_path="../Embeddings/", embed_filename="embeddings.pkl")
    svm = SVMClassifier(model_name="SVC")
    svm.find_best_model_random(feat_train=features, lab_train=labels)
    ml_classifier = MLClassifier(ml_model=svm.model, model_name=svm.name)

    ml_classifier.train_classifier(features=features, labels=labels)