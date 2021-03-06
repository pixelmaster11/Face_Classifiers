
import numpy as np
from sklearn.svm import LinearSVC, SVC
from Classifiers import ml_utils
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from Helper import utilities
from Classifiers.classifier import MLClassifier
from Classifiers.base_classifier import BaseClassifier



class SVMClassifier(BaseClassifier):

#########################################################################################################
    '''
        Initialize the classifier model
    '''
#########################################################################################################
    def __init__(self, model_name = "SVC"):

        super().__init__()

        self.args = self.parse_args()
        self.name = model_name

        if model_name == "SVC":
            self.model = SVC(gamma="scale", random_state=22)


        elif model_name == "LinearSVC":
            self.model = LinearSVC(random_state=22, C=1)

#########################################################################################################
#
# Find best parameters using random search
#
#########################################################################################################
    '''
    Params:
        @:param: feat_train - Training features
        @:param: lab_train - Training labels
    '''

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
                                           cv=4,
                                           verbose=1,
                                           random_state=22, n_jobs=-1)

        random_search.fit(feat_train, lab_train)

        print("\nThe best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        print("The mean accuracy of a model with these hyperparameters is:")
        print(random_search.best_score_)

        #self.model = random_search.best_estimator_


#########################################################################################################
#
# Find best parameters using a much deeper grid based search with parameters set around those
# found in random search
#
#########################################################################################################
    '''
    Params:
        @:param: feat_train - Training features
        @:param: lab_train - Training labels
    '''

    def find_best_model_grid(self, feat_train, lab_train,):

        print("\nFinding best model using grid search..")


        # Scale the input features
        feat_train = ml_utils.get_scaing(scaling_type="Norm").fit(feat_train).transform(feat_train)
        print("Features after Scaling: {}".format(np.array(feat_train).shape))

        # Create the parameter grid based on the results of random search
        C = [1, 10, 100]
        degree = [5,6,7]
        gamma = [0.01, 0.1, 1]
        probability = [True]


        param_grid = [
            {'C': C, 'kernel': ['linear'], 'probability': probability},
            {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
            {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
        ]

        # Get cross validation splits
        cv = ml_utils.get_spliting(n_splits=10, test_size=0.3, random_state=22)

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
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        # Assign the best model
        self.model = grid_search.best_estimator_

if __name__ == '__main__':

    # Create the classifier
    svm = SVMClassifier(model_name="SVC")

    # Load features and labels
    features, labels, ips = utilities.load_embeddings(load_path=svm.args["embed_load_dir"],
                                                      embed_filename= svm.args["embed_filename"])

    # Train using default params
    ml_classifier = MLClassifier(ml_model=svm.model, model_name=svm.name)
    ml_classifier.train_classifier(features=features, labels=labels, save_model=svm.args["save_model"], save_name=svm.args["model_filename"])

    # Find the best params
    svm.find_best_model_random(feat_train=features, lab_train=labels)
    svm.find_best_model_grid(feat_train=features, lab_train=labels)

    # Train with best found params
    ml_classifier.set_model(ml_model=svm.model)
    ml_classifier.train_classifier(features=features, labels=labels, save_model=svm.args["save_model"], save_name=svm.args["model_filename"])