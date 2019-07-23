
import os
import cv2
import argparse
from imutils import paths

'''
Helper functions for quick image access
'''

# Get images at given path
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


# Get all image file paths at given image directory
def get_imagepaths(image_dir):

    image_paths = []

    for ip in paths.list_images(image_dir):
        print(ip)
        image_paths.append(ip)

    print("Total paths found {}".format(len(image_paths)))
    return image_paths

# Bool Arguement parser
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')







