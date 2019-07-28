
from Helper import utilities
from Classifiers import ml_utils
import numpy as np
import time
from sklearn.svm import LinearSVC, SVC

class MLClassifier():

    def __init__(self, ml_model = None, model_name=""):

        self.name = model_name
        self.model = ml_model

    def load_model(self, filename, load_dir = "../MLModels"):

       self.model, self.name, labels_train, features_train = ml_utils.load_ml_model(load_dir=load_dir, filename=filename)


    def train_classifier(self, features, labels, scaling = "Norm", split = "Shuffle", decompose = False, dcomp = "TSNE",
                         save_model=False, save_dir="../MLModels", save_name = "", plot_cm = True):


        model = self.model

        print("\nTraining classifier %s ..." % (self.name))

        print(self.model.get_params())

        t1 = time.time()

        # Scale the input features
        features = ml_utils.get_scaing(scaling_type=scaling).fit(features).transform(features)
        print("Features after Scaling: {}".format(np.array(features).shape))

        # Reduce dimensionality of features
        if decompose:
            features = ml_utils.get_decomposition(dcomp_type=dcomp).fit_transform(features)
            print("Features after Decomposition {}" .format(np.array(features).shape))

        # Split train and test sets
        feat_train, lab_train, feat_test, lab_test = ml_utils.split_train_data(features=features, labels=labels,
                                                                               test_percent=0.33, random_state=22)

        # Get type of cross validation splitting
        cv = ml_utils.get_spliting(split, n_splits=10, test_size=0.3, random_state=22)

        scores = ml_utils.calculate_cross_validate_score(ml_model=model, features=feat_train, labels=lab_train, cv=cv,
                                                         verbose=0)
        print("\nCV scores {}".format(scores))
        print("CV Accuracy for %s : %0.2f (+/- %0.2f)" % (self.name, scores.mean(), scores.std() * 2))

        # Build the classifier model
        model = ml_utils.build_model(ml_model=model, features_train=feat_train, labels_train=lab_train)

        t2 = time.time()
        print("\nTime taken to train classifier {} is: {} seconds".format(self.name, (t2 - t1)))

        # Test on test set
        self.test_classifier(ml_model=model, test_features=feat_test, test_labels=lab_test, plot_cm=plot_cm)

        if save_model:

            # Pickle model
            ml_utils.save_ml_model(ml_model=model, ml_name=self.name + "_" + save_name, features=features, labels=labels, save_dir=save_dir)

            header = ['Model Name', "Scaling", "CV Split", "CV Accuracy", "Params", "Train Time (sec)"]
            log = [self.name, scaling, split, scores.mean(), model.get_params(), (t2-t1)]

            # Save train logs to csv
            ml_utils.save_ml_model_log(ml_model=model, ml_name=self.name + "_" + save_name, save_dir=save_dir, header=header, log=log)

    def test_classifier(self, test_features, test_labels, ml_model = None, scale = False, scaling = "Norm", plot_cm = True):

        if ml_model is None:
            ml_model = self.model

        if scale:
            # Scale the input features
            test_features = ml_utils.get_scaing(scaling_type=scaling).fit(test_features).transform(test_features)

        lab_pred = ml_utils.make_prediction(ml_model=ml_model, features_test=test_features)
        accuracy = ml_utils.calculate_accuracy(labels_test=test_labels, labels_predicted=lab_pred)

        print("\nTest set Accuracy {}:".format(accuracy))


        # Print the classification report for detailed summary
        cr = ml_utils.build_classification_report(labels_actual=test_labels, labels_predicted=lab_pred, classes=test_labels)
        print("\n" + cr)

        if plot_cm:
            ml_utils.plot_confusion_matrix(labels_actual=test_labels, labels_predicted=lab_pred, classes=test_labels)

if __name__ == '__main__':

    features, labels, ips = utilities.load_embeddings(load_path="../Embeddings/", embed_filename="embeddings_ethnicity.pkl")

    svc = SVC(C=1, kernel="poly", max_iter=-1, degree=7, probability=True, gamma="scale")
    ml_cl = MLClassifier(ml_model=svc, model_name="SVC")

    ml_cl.train_classifier(features=features,labels=labels, save_model=True, save_name="ethnicity_recog")
    #ml_cl.load_model(filename="SVC_gender_recog.pkl")
    #ml_cl.test_classifier(test_features=features,test_labels=labels)

