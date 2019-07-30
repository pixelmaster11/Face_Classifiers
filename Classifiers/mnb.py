
from sklearn.naive_bayes import MultinomialNB
from Classifiers.base_classifier import BaseClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from Helper import utilities
from Classifiers.classifier import MLClassifier
from Classifiers import ml_utils

class MultiNomNaiveBayes(BaseClassifier):

    def __init__(self, model_name = "MN_NaiveBayes"):

        super().__init__()

        self.args = self.parse_args()
        self.name = model_name
        self.model = MultinomialNB()


    def find_best_model_random(self, feat_train, lab_train):
        pass

    def find_best_model_grid(self, feat_train, lab_train, ):
        pass



if __name__ == '__main__':

    # Create the ml model
    mn_nb = MultiNomNaiveBayes()

    # Load features and labels
    features, labels, ips = utilities.load_embeddings(load_path = mn_nb.args["embed_load_dir"], embed_filename=mn_nb.args["embed_filename"])


    # Train with default params
    ml_classifier = MLClassifier(ml_model=mn_nb.model, model_name=mn_nb.name)

    # Train using MinMax scaling to remove -ve values as MN NM does not support -ve values
    # Make sure to remove -ve values before feeding them to classifier
    ml_classifier.train_classifier(features=features, labels=labels, save_model=mn_nb.args["save_model"],
                                   save_name=mn_nb.args["model_filename"], scaling="MinMax")

