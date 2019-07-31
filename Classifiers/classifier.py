
from Helper import utilities
from Classifiers import ml_utils
import numpy as np
import time
from sklearn.svm import LinearSVC, SVC
import argparse

'''
This class is responsible for training the provided ML Model.
Performs cross validation and / or testing with different test dataset or using train / test split 
'''
class MLClassifier():

#########################################################################################################
#
# Initialize the classifier with give model
#
#########################################################################################################

    def __init__(self, ml_model = None, model_name=""):

        self.name = model_name
        self.model = ml_model

#########################################################################################################
#
# Sets the ml model to the provided model
#
#########################################################################################################
    '''
    Params:
        @:param: ml_model - ML model to set
    '''
    def set_model(self, ml_model):
        self.model = ml_model


#########################################################################################################
#
# Sets the ml model name
#
#########################################################################################################
    '''
    Params:
        @:param: ml_model - ML model to set
    '''
    def set_model_name(self, ml_name):
        self.name = ml_name

#########################################################################################################
#
# Loads a saved ml clasifier model
#
#########################################################################################################

    '''
    Params:
        @:param: filename - Name of the saved ml classifier model file
        @:param: load_dir - Directory from where to load the ml model
    '''

    def load_model(self, filename, load_dir = "../MLModels"):

       self.model, self.name, labels_train, features_train = ml_utils.load_ml_model(load_dir=load_dir, filename=filename)


#########################################################################################################
#
# Trains the classifier model
#
#########################################################################################################

    '''
    Params:
        @:param: features - Input features
        @:param: labels - Corresponding input labels
        @:param: scaling - Type of scaling to be applied on the input features
        @:param: split - Type of splitting to be used for cross validation
        @:param: decompose - Whether to use PCA / TSNE decomposition or not
        @:param: dcomp - Type of decomposition to be used if any
        @:param: save_model - Whether to save this classifier model to file
        @:param: save_dir - Path where the ml model would be saved
        @:param: save_name - Filename to be used during save
        @:param: plot_cm - Whether to plot the confusion matrix
        @:param: test_percent - Percent of input data to be used as test data
    '''
    def train_classifier(self, features, labels, scaling = "Norm", split = "Shuffle", decompose = False, dcomp = "TSNE",
                         save_model=False, save_dir="../MLModels", save_name = "", plot_cm = True, test_percent = 0.33):

        # Assign the current ml model
        model = self.model

        print("\nTraining classifier %s using following parameters.." % (self.name))
        print(self.model.get_params())

        # Calculate time taken to train classifier
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
                                                                               test_percent=test_percent, random_state=22)

        # Get type of cross validation splitting
        cv = ml_utils.get_spliting(split, n_splits=10, test_size=0.3, random_state=22)

        # Get the CV scores
        scores = ml_utils.calculate_cross_validate_score(ml_model=model, features=features, labels=labels, cv=cv, verbose=0)

        print("\nCV scores {}".format(scores))
        print("CV Accuracy for %s : %0.2f (+/- %0.2f)" % (self.name, scores.mean(), scores.std() * 2))

        # Build the classifier model
        model = ml_utils.build_model(ml_model=model, features_train=feat_train, labels_train=lab_train)

        t2 = time.time()
        print("\nTime taken to train classifier {} is: {} seconds".format(self.name, (t2 - t1)))

        # Test on train set
        train_accuracy = self.test_classifier(ml_model=model, test_features=feat_train, test_labels=lab_train, plot_cm=False, title="Train Accuracy")

        # Test on test set
        test_accuracy = self.test_classifier(ml_model=model, test_features=feat_test, test_labels=lab_test, plot_cm=plot_cm)

        if save_model:

            # Pickle model
            ml_utils.save_ml_model(ml_model=model, ml_name=self.name + "_" + save_name, features=features, labels=labels, save_dir=save_dir)

            header = ['Model Name', "Scaling", "CV Split", "CV Accuracy", "Train Accuracy ", "Test Accuracy", "Params", "Train Time (sec)"]
            log = [self.name, scaling, split, scores.mean(), train_accuracy, test_accuracy, model.get_params(), (t2-t1)]

            # Save train logs to csv
            ml_utils.save_ml_model_log(ml_model=model, ml_name=self.name + "_" + save_name, save_dir=save_dir, header=header, log=log)

#########################################################################################################
#
# Test the model on a test set
#
#########################################################################################################

    '''
    Params:
        @:param: test_features - Features to test
        @:param: test_labels - Corresponding test labels
        @:param: ml_model - Classifier model
        @:param: scale - Whether to scale the features
        @:param: scaling - Type of scaling to apply if any
        @:param: plot_cm - Whether to plot the confusion matrix
    
    Returns:
        @:returns: accuracy - Returns the accuracy
    '''
    def test_classifier(self, test_features, test_labels, ml_model = None, scale = False, scaling = "Norm", plot_cm = True, title  = "Test Accuracy"):

        # Set the current model if provided model is none
        if ml_model is None:
            ml_model = self.model

        if scale:
            # Scale the input features
            test_features = ml_utils.get_scaing(scaling_type=scaling).fit(test_features).transform(test_features)

        lab_pred = ml_utils.make_prediction(ml_model=ml_model, features_test=test_features)
        accuracy = ml_utils.calculate_accuracy(labels_test=test_labels, labels_predicted=lab_pred)

        print("\n" + title + " {}:".format(accuracy))
        #print("\nTest set Accuracy {}:".format(accuracy))


        # Print the classification report for detailed summary
        cr = ml_utils.build_classification_report(labels_actual=test_labels, labels_predicted=lab_pred, classes=test_labels)
        print("\n" + cr)

        if plot_cm:
            ml_utils.plot_confusion_matrix(labels_actual=test_labels, labels_predicted=lab_pred, classes=test_labels)

        return accuracy



####################################################################################################################################
#
# Construct the argument parser and parse the arguments
#
###################################################################################################################################
def parse_args():


    ap = argparse.ArgumentParser()


    ap.add_argument("-edl",
                    "--embeddings_load_dir",
                    required=False,
                    help="Path where to load the saved embeddings file",
                    default="../Embeddings\\")

    ap.add_argument("-ef",
                    "--embed_filename",
                    required=False,
                    help="Name of embeddings file that will be saved or loaded from",
                    default="embeddings")

    ap.add_argument("-s",
                    "--save_model",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to save trained model")

    ap.add_argument("-mf",
                    "--model_filename",
                    required=False,
                    help="Name of the model file that will be saved or loaded from",
                    default="SVC_face_recog")

    ap.add_argument("-mds",
                    "--model_save_dir",
                    required=False,
                    help="Path where to save the model file",
                    default="../MLModels\\")


    ap.add_argument("-mdl",
                    "--model_load_dir",
                    required=False,
                    help="Path from where to load the saved model file",
                    default="../MLModels\\")



    ap.add_argument("-m",
                    "--mode",
                    choices=["load", "save"],
                    required=False,
                    help="Whether to load already existing model or generate and save new one",
                    default="save")

    return vars(ap.parse_args())





if __name__ == '__main__':

    # Get the arguements
    args = parse_args()

    # Load the features and labels
    features, labels, ips = utilities.load_embeddings(load_path=args["embeddings_load_dir"], embed_filename=args["embed_filename"])

    ml_cl = MLClassifier()

    # If train new model
    if args["mode"] == "save":
        svc = SVC(C=1, kernel="poly", max_iter=-1, degree=7, probability=True, gamma="scale")
        ml_cl.set_model(ml_model=svc)
        ml_cl.set_model_name(ml_name="SVC")
        ml_cl.train_classifier(features=features,labels=labels, save_model=args["save_model"], save_name=args["model_filename"])

    # Load already existing model and train
    else:
        ml_cl.load_model(filename="SVC_gender_recog.pkl")
        ml_cl.train_classifier(features=features, labels=labels, save_model=args["save_model"], save_name=args["model_filename"])
        #ml_cl.test_classifier(test_features=features,test_labels=labels)

