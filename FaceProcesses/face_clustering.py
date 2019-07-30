
import argparse
import cv2
import dlib
import os
import numpy as np
from FaceProcesses.face_encodings import Face_Encoding
from Helper.generate_dataset import GenerateDataset
from Helper import utilities
from sklearn.cluster import DBSCAN


'''
This class is responsible for clustering of face images
'''


class FaceClustering:

#########################################################################################################
#
# Initialize
#
#########################################################################################################

    def __init__(self):
        pass


#########################################################################################################
#
# This function creates clusters based on provided vectors
#
#########################################################################################################
    '''
    Params:
        @:param: - descriptors - Computed Face image descriptors / vector representations
        @:param - d_value - Distance value used during cluster computation
        @:param - method - Type of clustering method: Chinese whispers or DBScan
    
    Returns:
        @:returns: clusters - Computed Clusters
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

#########################################################################################################
#
# This function finds and returns the biggest cluster number from the given clusters
#
#########################################################################################################
    '''
    Params:
        @:param: clusters - Created clusters
    
    Returns:
        @:returns: biggest_cluster - Index of biggest_cluster
    '''
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

#########################################################################################################
#
# Get all the indices of images corresponding to the biggest cluster
#
#########################################################################################################

    '''
       Params:
           @:param: clusters - Created clusters
           @:param: biggest_cluster - Biggest cluster index
    
       Returns:
           @:returns: biggest_cluster - Indices of images belonging to the biggest_cluster
    '''
    def get_indices_of_biggest_cluster(self, clusters, biggest_cluster):

        # Find the indices for the biggest class
        indices = []
        for i, label in enumerate(clusters):
            if label == biggest_cluster:
                indices.append(i)

        print("Indices of images in the biggest cluster: {}".format(str(indices)))

        return indices

#########################################################################################################
#
#   Display images belonging to the biggest cluster
#
#########################################################################################################
    def display_biggest_cluster_images(self, biggest_cluster_indices, image_paths, cluster_images, read_images = False):
        for i, index in enumerate(biggest_cluster_indices):
                if read_images:
                    img = cv2.imread(image_paths[index], cv2.IMREAD_UNCHANGED)
                else:
                    img = cluster_images[index]

                    # Scale large images when using GPU
                    if (img.shape[1] > 1000 or img.shape[0] > 1000) and dlib.DLIB_USE_CUDA:
                        resize = True
                    else:
                        resize = False

                    if resize:
                        scale_percent = 50  # percent of original size
                        width = int(img.shape[1] * scale_percent / 100)
                        height = int(img.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        # resize image
                        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow("Window", img)
                cv2.waitKey(0)

        cv2.destroyWindow("Window")


#########################################################################################################
#
# Displays all images in the formed clusters
#
#########################################################################################################

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

                        r = img.shape[1] / float(img.shape[1])


                        # Scale large images when using GPU and if input images were not alligned
                        if (img.shape[1] > 1000 or img.shape[0] > 1000) and dlib.DLIB_USE_CUDA:
                            resize = True
                        else:
                            resize = False

                        if resize:
                            scale_percent = 50  # percent of original size
                            width = int(img.shape[1] * scale_percent / 100)
                            height = int(img.shape[0] * scale_percent / 100)
                            dim = (width, height)
                            # resize image
                            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)\

                    else:
                        img = cluster_images[i]


                    #img = cv2.resize(img, (320, 320))

                    if fdm == "CNN":
                        box = boxes[i].rect
                    else:
                        box = boxes[i]

                    x = int(box.left() * r)
                    y = int(box.top() * r)
                    w = int(box.right() * r) - x
                    h = int(box.bottom() * r) - y

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    if save_to_file:
                        p = os.path.join(save_path, ("Cluster_" + str(cluster_no) + "_" + str(i) + "." + image_paths[i].split(".")[-1]))
                        print(p)
                        cv2.imwrite(p, img)


                    else:
                        cv2.imshow("Cluster no %s"%str(cluster_no), img)
                        cv2.waitKey(0)
                        cv2.destroyWindow("Cluster no %s"%str(cluster_no))

        if save_to_file:
            print("Cluster images saved to file at path {}".format(save_path))



def parse_args():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-id",
                    "--image_dir",
                    required=True,
                    help="Path for the images to cluster")

    ap.add_argument("-m",
                    "--clustering_method",
                    choices=["CW", "DB"],
                    required=False,
                    help="Clustering algorithm to use either Chinese Whisper (CW) or DBScan (DB). Use DB if you have a small dataset",
                    default="CW")

    ap.add_argument("-fd",
                    "--face_detection_method",
                    choices=["CNN", "HOG"],
                    required=False,
                    help="type of face detection to use either HOG or deep learning CNN",
                    default="CNN")

    ap.add_argument("-fl",
                    "--face_landmarks_method",
                    choices=["68", "5"],
                    required=False,
                    help="Whether to use a 68-point or 5-point based landmark detection model",
                    default="68")

    ap.add_argument("-mp",
                    "--multi_proc",
                    required=False,
                    default=False, type=utilities.str2bool, nargs='?',
                    help="Whether to use multiprocessing")

    ap.add_argument("-gpu",
                    "--use_gpu",
                    required=False,
                    default=True,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use GPU for computation")

    ap.add_argument("-eds",
                    "--embeddings_save_dir",
                    required=False,
                    help="Path where to save the generated embeddings",
                    default="../Embeddings\\")

    ap.add_argument("-edl",
                    "--embeddings_load_dir",
                    required=False,
                    help="Path where to load the saved embeddings from",
                    default="../Embeddings\\")

    ap.add_argument("-m",
                    "--mode",
                    choices=["load", "gen"],
                    required=False,
                    help="Whether to load already existing embeddings or generate and save new ones",
                    default="load")

    ap.add_argument("-s",
                    "--save_embeddings",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to save the computed cluster images embeddings")

    ap.add_argument("-ef",
                    "--embed_filename",
                    required=False,
                    help="Name of generated embeddings file to be saved",
                    default="embeddings_cluster.pkl")

    ap.add_argument("-a",
                    "--allign",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to allign images before computation")

    ap.add_argument("-wr",
                    "--write_cluster_images",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to write the cluster images to disk")

    ap.add_argument("-wp",
                    "--cluster_write_path",
                    required=False,
                    help="Path where to save the generated cluster images",
                    default="../Images\\Generated_Clusters")

    return vars(ap.parse_args())


if __name__ == '__main__':

    # Get the arguements
    args = parse_args()
    ip = args["image_dir"]
    gpu = args["use_gpu"]
    flm = args["face_landmarks_method"]
    fdm = args["face_detection_method"]
    embed_svdir = args["embeddings_save_dir"]
    embed_ldir = args["embeddings_load_dir"]
    save = args["save_embeddings"]
    filename = args["embed_filename"]
    write_clusters = args["write_cluster_images"]
    write_path_cluster = args["cluster_write_path"]
    allign = args["allign"]
    method = args["clustering_method"]
    mode = args["mode"]

    display_biggest = False

    gen = GenerateDataset(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu, verbose=1)

    fc = FaceClustering()
    cluster_images = []
    descriptors = []

    #If generate only embeddings (without labels)
    if mode == "gen":
        # Generate Embeddings
        embeddings, boxes , image_paths = gen.generate_only_emebddings_at_path(image_dir=ip, filename=filename, allign=allign, save=save, save_dir=embed_svdir)

    # Else load already generated only embeddings file (without labels)
    else:
        # Load generated embeddings file from given path
        embeddings, boxes, image_paths = gen.load_only_embeddings(filename=filename, load_dir=embed_ldir)

    # Convert generated embeddings into a list of dlib vectors which is required for clustering
    for e in embeddings:
        e = dlib.vector(e)
        print(type(e))
        descriptors.append(e)

    print(np.array(descriptors).shape)
    print(type(descriptors))

    # Store alligned images
    if allign:
        images = utilities.get_images(image_path=ip)

        for image in images:
            alligned_images = gen.fe.fd.get_alligned_face(image=image)

            for img in alligned_images:
                cluster_images.append(img)

    # Create clusters
    c = fc.create_clusters(descriptors=descriptors, method=method)

    # Find the biggest cluster
    bc = fc.find_biggest_cluster(clusters=c)

    # Get the biggest cluster indices
    bc_indices = fc.get_indices_of_biggest_cluster(clusters=c, biggest_cluster=bc)

    if display_biggest:
        # Display the biggest cluster images
        fc.display_biggest_cluster_images(biggest_cluster_indices=bc_indices, image_paths=image_paths, cluster_images=cluster_images, read_images=not allign)

    # Display all cluster images
    fc.display_cluster_images(clusters=c, image_paths=image_paths, cluster_images=cluster_images,read_images=not allign,save_to_file=write_clusters, save_path=write_path_cluster)



