
import argparse
import cv2
import dlib
import os
import numpy as np
from face_encodings import Face_Encoding
import similarity_metrics as sm
import utilities
from sklearn.cluster import DBSCAN
from face_allignment import FaceAlligner

'''
This class is responsible for clustering of face images
'''


class FaceClustering:

    # Initialize with a d_value which is just a distance / threshold value used for clustering
    def __init__(self):
        pass

    # This function creates clusters based on provided vectors
    '''
    Params:
        descriptors - Computed Face image descriptors / vector representations
    
    Returns:
        clusters - Computed Clusters
    '''
    def create_clusters(self, descriptors, d_value = 0.5, method = "CW"):


        print("\nUsing {} cluster method".format(method))

        if method == "DB":
            #Compute clusters using DBSCAN
            clust = DBSCAN(metric="euclidean", n_jobs=1)
            clust.fit(descriptors)
            clusters = np.unique(clust.labels_)
            num_classes = len(np.where(clusters > -1)[0])
            clusters = clust.labels_

        elif method == "CW":
            # Computer clusters using chinese whispers
            clusters = dlib.chinese_whispers_clustering(descriptors, d_value)
            num_classes = len(set(clusters))

        else:
            print("Please provide proper method as CW or DB")
            exit()


        print("Number of clusters: {}".format(num_classes))
        print("Clusters: {}".format(clusters))


        return clusters

    # This function finds and returns the biggest cluster number from the given clusters
    def find_biggest_cluster(self, clusters):

        # Find biggest class
        biggest_cluster = None
        biggest_cluster_length = 0
        num_classes = len(set(clusters))

        for i in range(0, num_classes):
            class_length = len([label for label in clusters if label == i])
            if class_length > biggest_cluster_length:
                biggest_cluster_length = class_length
                biggest_cluster = i

        print("\nBiggest cluster id number: {}".format(biggest_cluster))
        print("Number of faces in biggest cluster: {}".format(biggest_cluster_length))

        return biggest_cluster

    # Get all the indices of images corresponding to the biggest cluster
    def get_indices_of_biggest_cluster(self, clusters, biggest_cluster):

        # Find the indices for the biggest class
        indices = []
        for i, label in enumerate(clusters):
            if label == biggest_cluster:
                indices.append(i)

        print("Indices of images in the biggest cluster: {}".format(str(indices)))

        return indices


    def display_biggest_cluster_images(self, biggest_cluster_indices, image_paths, cluster_images, read_images = False):
        for i, index in enumerate(biggest_cluster_indices):
                if read_images:
                    img = cv2.imread(image_paths[index], cv2.IMREAD_UNCHANGED)
                else:
                    img = cluster_images[index]
                img = cv2.resize(img, (320,320))
                cv2.imshow("Window", img)
                cv2.waitKey(0)

        cv2.destroyWindow("Window")

    # Displays all images in the given cluster
    def display_cluster_images(self, clusters, image_paths, cluster_images, read_images = False, save_to_file = False, save_path = "Cluster\\"):


        if save_to_file:
            # Ensure output directory exists
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        num_classes = len(set(clusters))
        print(num_classes)

        for num in range(-1, num_classes):

            for i, cluster_no in enumerate(clusters):

                if cluster_no == num:
                    if read_images:
                        img = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
                    else:
                        img = cluster_images[i]
                    #img = cv2.resize(img, (320, 320))

                    if save_to_file:
                        p = os.path.join(save_path, str(cluster_no) + "_" + image_paths[i])
                        cv2.imwrite(p, img)

                    else:
                        cv2.imshow("Cluster no %s"%str(cluster_no), img)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Cluster no %s"%str(cluster_no))





def parse_args():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-id", "--image_dir", required=True, help="Path for the images")
    ap.add_argument("-fd", "--face_detection", choices=["CNN", "HOG"], required=False,
                    help="type of face detection using HOG or deep learning CNN", default="CNN")

    ap.add_argument("-fl", "--face_landmarks", choices=["68", "5"], required=False,
                    help="Whether to use a 68-point or 5-point based landmark detection model", default="68")

    ap.add_argument("-mp", "--multi_proc", required=False, default=True, type=utilities.str2bool, nargs='?',
                    help="Whether to use multiprocessing")

    return vars(ap.parse_args())


if __name__ == '__main__':

    descriptors = []
    fe = Face_Encoding(face_detection_model="HOG", face_landmark_model="68")
    fc = FaceClustering()
    cluster_images = []
    ip = "Images\\Face_Clustering\\"



    image_paths = utilities.get_imagepaths(ip)
    #image_list = utilities.get_images(image_path=ip)


    for image_path in image_paths:

        print("\n"+image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (500, 500))

        #dets = fe.fd.detect_face(image=image)

        #if len(dets) > 0:
        #    image =  fe.fd.get_alligned_face(image=image, dets=dets)[0]

        # Extract the 128-D face embedding
        embeddings, drawn_image = fe.compute_facenet_embedding_dlib(image=image, draw=True)


        if embeddings is None:
            print("Could not extract descriptor")
            continue

        elif len(embeddings) == 0:
            print("Could not extract descriptor")
            continue

        # For all embeddings returned in case of multiple faces in a single image
        # Append all embeddings into a list and append its corresponding image
        for e in embeddings:

            if e is None:
                continue

            elif len(e) == 0:
                continue

            descriptors.append(e)
            cluster_images.append(drawn_image)
        print(np.array(descriptors).shape)




    c = fc.create_clusters(descriptors=descriptors, method="CW")
    bc = fc.find_biggest_cluster(clusters=c)
    bc_indices = fc.get_indices_of_biggest_cluster(clusters=c, biggest_cluster=bc)
    fc.display_biggest_cluster_images(biggest_cluster_indices=bc_indices, image_paths=image_paths, cluster_images=cluster_images)
    fc.display_cluster_images(clusters=c, image_paths=image_paths, cluster_images=cluster_images)



