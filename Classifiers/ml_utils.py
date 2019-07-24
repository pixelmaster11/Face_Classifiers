from sklearn.preprocessing import StandardScaler, Binarizer, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold, cross_val_score, \
                                    ShuffleSplit, StratifiedKFold, LeaveOneOut

from sklearn import metrics
from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# This function returns a scaling function of specified type
def get_scaing(scaling_type = "Std"):

    if scaling_type == "Norm":
        return Normalizer()

    elif scaling_type == "Std":
        return StandardScaler()

    elif scaling_type == "MinMax":
        return MinMaxScaler()

    elif scaling_type == "MaxAbs":
        return MaxAbsScaler()

    elif scaling_type == "Robust":
        return RobustScaler()

    elif scaling_type == "Bin":
        return Binarizer()

    else:
        print("Please enter a valid scaling type from Std, Norm, MinMax, MaxAbs, Robust or Bin..\nDefaulting to Std..")
        return StandardScaler()


# This function returns the type of dimensionality reduction technique used
def get_decomposition(dcomp_type = "TSNE", n_components = 2, neighbours = 5):

    if dcomp_type == "TSNE":
        mod = TSNE(n_components=n_components, perplexity=neighbours, early_exaggeration=12.0,
                   learning_rate=200.0, n_iter=1000, n_iter_without_progress=300,
                   min_grad_norm=1e-07, metric="euclidean",
                   init="random", verbose=0, random_state=None, method="barnes_hut", angle=0.5)

    elif dcomp_type == "PCA":
        mod = PCA(n_components=n_components)

    else:
        mod = PCA(n_components=n_components)


    return mod

# This function returns the type of splitting used for cross-validation
def get_spliting(spliting_type = "Shuffle", use_shuffling = True, random_state = 22, test_size = 0.2, n_splits = 10):

    if spliting_type == "Shuffle":
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    elif spliting_type == "Kfold":
        cv = KFold(n_splits=n_splits, random_state=22, shuffle=use_shuffling)

    elif spliting_type == "SKfold":
        cv = StratifiedKFold(n_splits=n_splits, random_state=22, shuffle=use_shuffling)

    elif spliting_type == "SShuffle":
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=22)

    elif spliting_type == "LOO":
        cv = LeaveOneOut()

    else:
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=22)

    return cv


# This functions splits the input data into train and test sets and returns it
def split_train_data(features, labels, test_percent = 0.33, random_state = 22):
    feat_train, feat_test, lab_train, lab_test = train_test_split(features, labels, test_size=test_percent, random_state=random_state)

    return feat_train,lab_train, feat_test, lab_test

# This function calculates the cross-validation scores and returns it
def calculate_cross_validate_score(ml_model, features, labels, cv, scoring = "accuracy", n_jobs = -1, verbose = 0):

    scores = cross_val_score(ml_model, features, labels, cv=cv, scoring=scoring, n_jobs=n_jobs)

    if verbose == 1:
        print(scores)
        print("CV Accuracy for %s : %0.2f (+/- %0.2f)" % (ml_model, scores.mean(), scores.std() * 2))

    return scores

# This function calculates accuracy from the predicted label values
def calculate_accuracy(labels_test, labels_predicted):

    accuracy = metrics.accuracy_score(labels_test, labels_predicted)
    return accuracy


# This function builds the given ml model and fits it with train features and labels and tests it on test features

def build_model(ml_model, features_train, labels_train, features_test = [], predict = True):

    ml_model.fit(features_train, labels_train)

    if predict:
        if len(features_test) == 0:
            print("Please provide test features")
            exit()

        labels_predicted = ml_model.predict(features_test)
        return labels_predicted

    else:
        return ml_model



def build_confusion_matrix(labels_actual, labels_predicted):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(labels_actual, labels_predicted)

    return cm


def build_classification_report(labels_actual, labels_predicted, classes):
    classes = sorted(list(set(classes)))
    cr = metrics.classification_report(labels_actual, labels_predicted, target_names=classes)
    return cr

######################################################################################
#
# Plots the confusion matrix
#
######################################################################################
def plot_confusion_matrix(labels_actual, labels_predicted, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = build_confusion_matrix(labels_actual=labels_actual, labels_predicted=labels_predicted)

    # Only use the labels that appear in the data
    classes = sorted(list(set(classes)))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #return ax
    plt.show()


# Saves the given ml model to file
def save_ml_model(ml_model, features, labels, save_dir = "MLModels\\"):

    filename = ml_model + ".pkl"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + filename, 'wb') as outfile:
        pickle.dump((ml_model, labels, features), outfile)
    print('Saved Best classifier model {} to file {}'.format(ml_model, save_dir + filename))

# Loads the ml model from file
def load_ml_model(load_dir = "MLModels\\", filename = ""):

    with open(load_dir + filename, 'rb') as infile:
        (ml_model, features, labels) = pickle.load(infile)

    print('Loaded classifier model %s from file "%s"' %(ml_model, load_dir+filename) )

    return ml_model, features, labels

