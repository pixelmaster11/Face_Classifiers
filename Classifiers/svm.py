
import time
import numpy as np
from sklearn.svm import LinearSVC, SVC
from Classifiers import ml_utils
import utilities


class SVMClassifier():

    def __init__(self, model_name = "SVC"):

        self.features = []
        self.labels = []
        self.name = ""

        if model_name == "SVC":
            self.model = SVC(gamma="scale", kernel="rbf", C=1)
            self.name = "SVC"

        elif model_name == "LinearSVC":
            self.model = LinearSVC(random_state=22, C=1)
            self.name = "LinearSVC"




        #[SVC(kernel='rbf', probability=True, C=0.2, random_state=22, max_iter=-1, gamma="scale",
        #     decision_function_shape='ovr', shrinking=True,
        #     degree=10, tol=0.001),
        # LinearSVC(C = 1, random_state=22, max_iter=10000, fit_intercept=True, dual=True, intercept_scaling=1, loss='squared_hinge',
        #         multi_class='ovr', penalty='l2', tol=0.0001),

    def load_features(self, feature_dir, feature_filename):
        self.features, self.labels, dataset_imagepaths =  utilities.load_embeddings(load_path=feature_dir, embed_filename=feature_filename)

        feats = np.empty((len(self.labels), 128))

        for i,feat in enumerate(self.features):

            feat = np.array(feat).reshape(1,-1)
            feats[i] = feat


        self.features = feats
        print("Loaded features and labels successfully %s %s" % ((np.array(self.features).shape), np.array(self.labels).shape))

    def train_classifier(self):

        print("\nTraining classifier..")

        features = self.features
        model = self.model
        labels = self.labels

        t1 = time.time()

        # Scale the input features
        features = ml_utils.get_scaing(scaling_type="Norm").fit(features).transform(features)

        print("Features after Scaling: {}" .format(np.array(features).shape))

        # Reduce dimensionality of features
        #features = ml_utils.get_decomposition(dcomp_type="PCA").fit_transform(features)
        print("Features after Decomposition {}" .format(np.array(features).shape))

        # Split train and test sets
        # 80% training and 20% test
        feat_train, lab_train, feat_test, lab_test = ml_utils.split_train_data(features=features, labels=labels, test_percent=0.4)

        # Get type of cross validation splitting
        cv = ml_utils.get_spliting("Kfold",n_splits=10)

        scores = ml_utils.calculate_cross_validate_score(ml_model=model, features=feat_train, labels=lab_train, cv = cv, verbose=0)
        print("\nCV scores {}".format(scores))
        print("CV Accuracy for %s : %0.2f (+/- %0.2f)" % (self.name, scores.mean(), scores.std() * 2))


        # Test on test set

        lab_pred = ml_utils.build_model(ml_model=model, features_train=feat_train, labels_train=lab_train, features_test=feat_test, predict=True)
        accuracy = ml_utils.calculate_accuracy(labels_test=lab_test, labels_predicted=lab_pred)
        print("\nAccuracy {}:".format(accuracy))

        t2 = time.time()
        print("\nTime taken to train classifier {} is: {} seconds".format(self.name, (t2 - t1)))

        # Print the classification report for detailed summary
        cr = ml_utils.build_classification_report(labels_actual=lab_test, labels_predicted=lab_pred, classes=labels)
        print("\n"+cr)

        plot_cm = True

        if plot_cm:
            ml_utils.plot_confusion_matrix(labels_actual=lab_test, labels_predicted=lab_pred, classes=labels)

        save_model = False

        if save_model:
            ml_utils.save_ml_model(ml_model=model, features=features, labels=labels)



if __name__ == '__main__':


    svm = SVMClassifier()
    svm.load_features(feature_dir="../Embeddings/",feature_filename="embeddings.pkl")
    svm.train_classifier()