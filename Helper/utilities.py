
import os
import cv2
import argparse
from imutils import paths
import numpy as np
import pickle

'''
Helper functions for quick image access
'''

#########################################################################################################
#
# Get images at given path
#
#########################################################################################################

def get_images(image_path, grayscale = False):

    images = []
    for ip in paths.list_images(image_path):
        if grayscale:
            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(ip)

        images.append(img)

    print("\nTotal images found {}".format(len(images)))
    return images

#########################################################################################################
#
# Get all image file paths at given image directory
#
#########################################################################################################
def get_imagepaths(image_dir):

    image_paths = []

    for ip in paths.list_images(image_dir):
        print(ip)
        image_paths.append(ip)

    print("Total paths found {}".format(len(image_paths)))
    return image_paths

#########################################################################################################
#
# Bool Arguement parser
#
#########################################################################################################
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



#########################################################################################################
#
# This function saves the given embeddings / labels and image_paths to the file
#
#########################################################################################################

'''
Params:

    @:param: embeddings - A list of embeddings to save
    @:param: labels - Corresponding list of labels
    @:param: image_paths - Corresponding list of image_paths
    @:param: save_path - Where to save the file
    @:param: embed_filename - Name of the generated save file
    
'''
def save_embeddings(embeddings, labels, image_paths, save_path="../Embeddings\\", embed_filename="embeddings"):

    print("Total features {}".format(np.array(embeddings).shape))

    # Create directory if it not exists
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not embed_filename.__contains__(".pkl"):
        output_path_embed = os.path.join(save_path, embed_filename) + ".pkl"
    else:
        output_path_embed = os.path.join(save_path, embed_filename)

    # Save features to file
    print('Saved embeddings to file as {}'.format(output_path_embed))


    data = embeddings, labels, image_paths


    with open(output_path_embed, 'wb') as outfile:
        pickle.dump(data, outfile)


#########################################################################################################
#
# This function saves the given embeddings / labels and image_paths to the file
# This is useful when calculating embeddings for making a prediction, where we do not have the ground truth / labels
#
#########################################################################################################
    '''
    Params:
        @:param: embeddings - A list of embeddings to save
        @:param: image_paths - Corresponding list of image_paths
        @:param: boxes - A list of detected face bounding box
        @:param: save_path - Where to save the file
        @:param: embed_filename - Name of the generated save file
    '''

def save_only_embeddings(embeddings, image_paths, boxes, save_path="../Embeddings\\", embed_filename="only_embeddings"):

    print("Total features {}".format(np.array(embeddings).shape))

    # Create directory if it not exists
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if not embed_filename.__contains__(".pkl"):
        output_path_embed = os.path.join(save_path, embed_filename) + ".pkl"
    else:
        output_path_embed = os.path.join(save_path, embed_filename)

    # Save features to file
    print('Saved embeddings, boxes, image_paths to file as {}'.format(output_path_embed))

    data = embeddings, boxes, image_paths

    with open(output_path_embed, 'wb') as outfile:
        pickle.dump(data, outfile)


#########################################################################################################
#
# Loads embedding file from given load_path
#
#########################################################################################################
    '''
    Params:
        @:param: load_path - From where to load the embeddings file
        @:param: embed_filename - Name of the file to be loaded

    Returns:
        @:returns: A tuple of (embeddings list, labels list, image_paths list)
    '''
def load_embeddings(load_path="../Embeddings\\", embed_filename="embeddings.pkl"):

    if not embed_filename.__contains__(".pkl"):
        embed_filename += ".pkl"

    # Loading Features
    with open(os.path.join(load_path, embed_filename), "rb") as infile:
        (dataset_embeddings, dataset_labels, dataset_imagepaths) = pickle.load(infile)

    print("\nLoaded embeddings file from {}".format(load_path + embed_filename))

    '''feats = np.empty((len(dataset_labels), 128))

    for i, feat in enumerate(dataset_embeddings):
        feat = np.array(feat).reshape(1, -1)
        feats[i] = feat'''

    print("Loaded features and labels successfully %s %s" % ((np.array(dataset_embeddings).shape), np.array(dataset_labels).shape))


    return dataset_embeddings, dataset_labels, dataset_imagepaths


#########################################################################################################
#
# Loads only embeddings file from given load_path
#
#########################################################################################################

'''
Params:
    @:param: load_path - From where to load the embeddings file
    @:param: embed_filename - Name of the file to be loaded

Returns:
    @:returns: A tuple of (embeddings list, boxes list, image_paths list)
'''

def load_only_embeddings(load_path="../Embeddings\\", embed_filename="embeddings.pkl"):


    if not embed_filename.__contains__(".pkl"):
        embed_filename += ".pkl"

    # Loading Features
    with open(os.path.join(load_path, embed_filename), "rb") as infile:
        (dataset_embeddings, dataset_boxes, dataset_imagepaths) = pickle.load(infile)

    print("\nLoaded embeddings file from {}".format(load_path + embed_filename))

    feats = np.empty((len(dataset_imagepaths), 128))

    for i, feat in enumerate(dataset_embeddings):
        feat = np.array(feat).reshape(1, -1)
        feats[i] = feat

    print("Loaded embeddings, boxes and image_paths successfully %s %s %s " % (np.array(feats).shape,
                                                                               np.array(dataset_boxes).shape,
                                                                               np.array(dataset_imagepaths).shape))

    dataset_embeddings = feats

    return dataset_embeddings, dataset_boxes, dataset_imagepaths