
import os
import cv2
import argparse
import pickle
import numpy as np
from face_encodings import Face_Encoding

from imutils import paths



def get_images(image_path, grayscale = False):

    images = []
    [images.append(cv2.imread(os.path.join(image_path, ip), cv2.IMREAD_UNCHANGED)) if not grayscale
     else images.append(cv2.imread(os.path.join(image_path, ip), cv2.IMREAD_GRAYSCALE ))
     for ip in os.listdir(image_path)]

    print("Total images found {}".format(len(images)))
    return images


def get_imagepaths(image_path):

    image_paths = []

    for ip in paths.list_images(image_path):
        print(ip)
        image_paths.append(ip)

    print("Total paths found {}".format(len(image_paths)))
    return image_paths


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def get_embeddings(image, fe):

    descriptors = []

    # Extract the 128-D face embedding
    embeddings, drawn_image = fe.compute_facenet_embedding_dlib(image=image, draw=True)

    if embeddings is None:
        print("Could not extract descriptor")
        return None

    elif len(embeddings) == 0:
        print("Could not extract descriptor")
        return None

    # For all embeddings returned in case of multiple faces in a single image
    # Append all embeddings into a list and append its corresponding image
    for e in embeddings:

        if e is None:
            continue

        elif len(e) == 0:
            continue

        descriptors.append(e)

    return descriptors


def get_embeddings_at_path(image_path, allign = True, resize = False, save_to_file = True, save_path = "Embeddings\\",
                           filename = "embeddings", fe = None):

    if fe is None:
        fe = Face_Encoding(face_detection_model="HOG", face_landmark_model="68")

    print("\nCalculating embeddings...")
    embeddings = []
    alligned_images = []
    labels = []
    images = []
    image_paths = get_imagepaths(image_path)

    for ip in image_paths:

        print("\n" + ip)
        image = cv2.imread(ip, cv2.IMREAD_UNCHANGED)

        if resize:
            image = cv2.resize(image, (500, 500))

        if allign:
            dets = fe.fd.detect_face(image=image)

            if len(dets) > 0:
                alligned_images = fe.fd.get_alligned_face(image=image, dets=dets)


            for image in alligned_images:

                e = get_embeddings(image, fe)
                if e is None:
                    continue
                elif len(e) == 0:
                    continue

                embeddings.append(e)
                labels.append(ip.split(os.sep)[-2])
                images.append(ip)

        else:
            embeds = get_embeddings(image, fe)

            if embeds is None:
                continue

            # For multiple faces in single image
            for e in embeds:

                if e is None:
                    continue
                elif len(e) == 0:
                    continue

                embeddings.append(get_embeddings(image, fe=fe))
                labels.append(ip.split(os.sep)[-2])
                images.append(ip)

        print(np.array(embeddings).shape)
        print(np.array(labels).shape)
        print(np.array(images).shape)

    if save_to_file:
        save_embeddings(embeddings=embeddings, labels=labels, image_paths = images, save_path=save_path, embed_filename=filename)

    return embeddings, labels, image_paths


def save_embeddings(embeddings, labels, image_paths, save_path = "Embeddings\\", embed_filename = "embeddings", label_filename = "labels",
                    image_path_filename = "imagepaths", single_file = True):

    print("Total features {}".format(np.array(embeddings).shape))

    # Create directory if it not exists
    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    output_path_embed = os.path.join(save_path, embed_filename) + ".pkl"

    # Save features to file
    print('Saved embeddings to file as {}'.format(output_path_embed))

    if single_file:
        data = embeddings, labels, image_paths

    else:
        data = embeddings

    with open(output_path_embed, 'wb') as outfile:
        pickle.dump(data, outfile)

    if not single_file:
        output_path_label = os.path.join(save_path, label_filename) + ".pkl"

        # Save features to file
        print('Saved labels to file as {}'.format(output_path_label))

        with open(output_path_label, 'wb') as outfile:
            pickle.dump(labels, outfile)

        output_path_images = os.path.join(save_path, image_path_filename) + ".pkl"

        # Save features to file
        print('Saved labels to file as {}'.format(output_path_images))

        with open(output_path_images, 'wb') as outfile:
            pickle.dump(image_paths, outfile)