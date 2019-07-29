
from sklearn.ensemble import GradientBoostingClassifier
from Classifiers.base_classifier import BaseClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from Helper import utilities
from Classifiers.classifier import MLClassifier
from Classifiers import ml_utils

class GradientBoostMethod(BaseClassifier):

    def __init__(self, model_name = "GradientBoost"):

        super().__init__()

        self.args = self.parse_args()
        self.name = model_name
        self.model = GradientBoostingClassifier(random_state=22)


    def find_best_model_random(self, feat_train, lab_train):

        # n_estimators
        n_estimators = [200, 800]

        # max_features
        max_features = ['auto', 'sqrt']

        # max_depth
        max_depth = [10, 40]
        max_depth.append(None)

        # min_samples_split
        min_samples_split = [10, 30, 50]

        # min_samples_leaf
        min_samples_leaf = [1, 2, 4]

        # learning rate
        learning_rate = [.1, .5]

        # subsample
        subsample = [.5, 1.]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'learning_rate': learning_rate,
                       'subsample': subsample}

        # Definition of the random search
        random_search = RandomizedSearchCV(estimator=self.model,
                                           param_distributions=random_grid,
                                           n_iter=50,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=1,
                                           random_state=8,
                                           n_jobs=-1)

        # Fit the random search model
        random_search.fit(feat_train, lab_train)

        print("The best hyperparameters from Random Search are:")
        print(random_search.best_params_)
        print("The mean accuracy of a model with these hyperparameters is:")
        print(random_search.best_score_)


    def find_best_model_grid(self, feat_train, lab_train):

        # Create the parameter grid based on the results of random search
        max_depth = [2, 3, 4, 5, 10]
        max_features = ['sqrt']
        min_samples_leaf = [1, 2]
        min_samples_split = [2, 5, 10]
        n_estimators = [800]
        learning_rate = [0.1, 0.5]
        subsample = [1.0]

        param_grid = {
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'subsample': subsample

        }



        # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
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

if __name__ == '__main__':

    # Create the classifier
    gbm = GradientBoostMethod()

    # Load features and labels
    features, labels, ips = utilities.load_embeddings(load_path=gbm.args["embed_load_dir"],
                                                      embed_filename= gbm.args["embed_filename"])

    # Train using default params
    ml_classifier = MLClassifier(ml_model=gbm.model, model_name=gbm.name)
    ml_classifier.train_classifier(features=features, labels=labels, save_model=gbm.args["save_model"], save_name=gbm.args["model_filename"])

    # Find the best params
    gbm.find_best_model_random(feat_train=features, lab_train=labels)
    gbm.find_best_model_grid(feat_train=features, lab_train=labels)

    # Train with best found params
    ml_classifier.set_model(ml_model=gbm.model)
    ml_classifier.train_classifier(features=features, labels=labels, save_model=gbm.args["save_model"], save_name=gbm.args["model_filename"])