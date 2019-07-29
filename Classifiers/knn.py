from Helper import utilities
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Classifiers import ml_utils
from Classifiers.classifier import MLClassifier
from Classifiers.base_classifier import BaseClassifier
from sklearn.model_selection import GridSearchCV

class KNNClassifier(BaseClassifier):

#########################################################################################################
#
# Initiallize the ml model
#
#########################################################################################################

    def __init__(self, model_name="KNN"):

        super().__init__()
        self.name = model_name
        self.model = KNeighborsClassifier()

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
        pass

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
    def find_best_model_grid(self, feat_train, lab_train):

        print("\nFinding best model using grid search..")

        # Scale the input features
        feat_train = ml_utils.get_scaing(scaling_type="Norm").fit(feat_train).transform(feat_train)
        print("Features after Scaling: {}".format(np.array(feat_train).shape))

        # Create the parameter grid
        n_neighbors = [int(x) for x in np.linspace(start=1, stop=100, num=100)]
        leaf_size = [1, 2, 3, 4, 5, 10, 15, 20]

        param_grid = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size}

        # Create cross validation splits
        cv = ml_utils.get_spliting(n_splits=3, test_size=0.33, random_state=22)

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=self.model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv,
                                   verbose=1,n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(feat_train, lab_train)

        print("\nThe best hyperparameters from Grid Search are:")
        print(grid_search.best_params_)
        print("\nThe mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        # Assign the best estimator model
        self.model = grid_search.best_estimator_

if __name__ == '__main__':

    # Create the ml model
    knn = KNNClassifier()

    # Load features and labels
    features, labels, ips = utilities.load_embeddings(load_path = knn.args["embed_load_dir"], embed_filename=knn.args["embed_filename"])

    # Train with default params
    ml_classifier = MLClassifier(ml_model=knn.model, model_name=knn.name)
    ml_classifier.train_classifier(features=features, labels=labels, save_model=knn.args["save_model"],
                                   save_name=knn.args["model_filename"])

    # Find best params
    knn.find_best_model_grid(feat_train=features,lab_train=labels)

    # Train with best params
    ml_classifier.train_classifier(features=features, labels=labels, save_model=knn.args["save_model"],
                                   save_name=knn.args["model_filename"])
