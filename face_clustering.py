
from imutils import paths
import cv2
import dlib
import os
import numpy as np
from face_encodings import Face_Encoding

class FaceClustering:

    def __init__(self, d_value = 0.5):
        self.d_value = d_value


    def create_clusters(self, descriptors):

        clusters = dlib.chinese_whispers_clustering(descriptors, self.d_value)
        num_classes = len(set(clusters))
        print("Number of clusters: {}".format(num_classes))
        print("Clusters: {}".format(clusters))

        return clusters

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

        print("Biggest cluster id number: {}".format(biggest_cluster))
        print("Number of faces in biggest cluster: {}".format(biggest_cluster_length))

        return biggest_cluster

    def get_indices_of_biggest_cluster(self, clusters, biggest_cluster):

        # Find the indices for the biggest class
        indices = []
        for i, label in enumerate(clusters):
            if label == biggest_cluster:
                indices.append(i)

        print("Indices of images in the biggest cluster: {}".format(str(indices)))

        return indices

    def display_biggest_cluster_images(self, biggest_cluster_indices, image_paths):
        for i, index in enumerate(biggest_cluster_indices):
                img = cv2.imread(image_paths[index], cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (320,320))
                cv2.imshow("Window", img)
                cv2.waitKey(0)

    def display_all_cluster_images(self, clusters, image_paths, cluster_images, read_images = False, save_to_file = False, save_path = ""):

        num_classes = len(set(clusters))

        if save_to_file:
            # Ensure output directory exists
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

        for num in range(0, num_classes):
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
                        cv2.imshow("Window", img)
                        cv2.waitKey(0)



if __name__ == '__main__':

    descriptors = []
    fe = Face_Encoding()
    fc = FaceClustering()

    image_paths = []
    cluster_images = []

    for image_path in paths.list_images("D:\Tuts\DataScience\Python\Datasets\FGNET\Age_Test\Old"):
        image_paths.append(image_path)

    for image_path in image_paths:

        input_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #input_image = cv2.resize(input_image, (320, 320))

        # Extract the 128-D face embedding
        embeddings, img = fe.compute_facenet_embedding_dlib(image=input_image, draw=True)

        if embeddings is None:
            print("Could not extract descriptor")
            continue

        elif len(embeddings) == 0:
            print("Could not extract descriptor")
            continue


        for e in embeddings:

            if e is None:
                continue

            elif len(e) == 0:
                continue

            descriptors.append(e)
            cluster_images.append(img)



        print(image_path)
        print(np.array(descriptors).shape)


    c = fc.create_clusters(descriptors=descriptors)
    bc = fc.find_biggest_cluster(clusters=c)
    bc_indices = fc.get_indices_of_biggest_cluster(clusters=c, biggest_cluster=bc)
    fc.display_all_cluster_images(clusters=c, image_paths=image_paths, cluster_images=cluster_images)



