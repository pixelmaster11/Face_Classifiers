import utilities
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Classifiers import ml_utils
from Classifiers.classifier import MLClassifier
from sklearn.model_selection import GridSearchCV

class KNNClassifier():

    def __init__(self, model_name="KNN"):

        self.name = model_name
        self.model = KNeighborsClassifier(leaf_size=1, n_neighbors=7)

    def find_best_model_grid(self, feat_train, lab_train):

        print("\nFinding best model using grid search..")

        # Scale the input features
        feat_train = ml_utils.get_scaing(scaling_type="Norm").fit(feat_train).transform(feat_train)

        print("Features after Scaling: {}".format(np.array(feat_train).shape))

        # Reduce dimensionality of features
        # features = ml_utils.get_decomposition(dcomp_type="TSNE").fit_transform(features)
        # print("Features after Decomposition {}" .format(np.array(features).shape))

        # Create the parameter grid based
        n_neighbors = [int(x) for x in np.linspace(start=1, stop=100, num=100)]
        leaf_size = [1, 2, 3, 4, 5, 10, 15, 20]

        param_grid = {'n_neighbors': n_neighbors, 'leaf_size': leaf_size}

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
        print("")
        print("The mean accuracy of a model with these hyperparameters is:")
        print(grid_search.best_score_)

        self.model = grid_search.best_estimator_

if __name__ == '__main__':

    features, labels, ips = utilities.load_embeddings(load_path ="../Embeddings/", embed_filename="embeddings.pkl")
    knn = KNNClassifier()
    knn.find_best_model_grid(feat_train=features,lab_train=labels)
    ml_classifier = MLClassifier(ml_model=knn.model, model_name=knn.name)

    ml_classifier.train_classifier(features=features, labels=labels)
