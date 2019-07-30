
from FaceProcesses.face_encodings import Face_Encoding
from Helper import utilities
import numpy as np
import argparse

class GenerateDataset():

##################################################################################################################################
    '''
    Params:
        @:param: face_detetion_model - Type of face detection to use CNN or HOG
        @:param: face_landmark_model - Type of face landmark detetction to use 68-point or 5-point based
        @:param: use_gpu - Whether to use GPU during computation
    '''
##################################################################################################################################

    def __init__(self, face_detection_model="CNN", face_landmark_model="68", use_gpu=True, verbose = 1):

        self.fe = Face_Encoding(face_detection_model=face_detection_model, face_landmark_model=face_landmark_model,
                                use_gpu=use_gpu, verbose=verbose)



###################################################################################################################################
#
# This function generates an embedding file for all the given images at the image_path
#
###################################################################################################################################
    '''
    Params:
        @:param: image_dir- Path for all the images of which to calculate and save embeddings
        @:param: allign - Whether to allign images or not
        @:param: resize - Whether to resize images
        @:param: save - Whether to save generated embeddings to file
        @:param: filename - Embbedings save filename
        @:param: save_dir - Path where the generated embeddings would be saved
    
    Returns:
        @:returns: A tuple of (np array embeddings, labels, image_paths)
        
    '''
    def generate_dataset(self, image_dir, filename, save_dir="../Embeddings", allign=False, resize=False,save=True):

        dataset_embeddings, \
        dataset_labels, \
        dataset_imagepaths = self.fe.get_embeddings_at_path(image_path=image_dir, allign=allign,
                                                                 resize=resize, save_path=save_dir, save_to_file=save,
                                                                 filename=filename)

        return dataset_embeddings, dataset_labels, dataset_imagepaths

###################################################################################################################################
#
#   This function calculates only the embeddings and not the labels.
#   This is useful for calculating embeddings for test images where we don't have the ground truths available
#
###################################################################################################################################
    '''
    Params:
        @:param: image_dir - Path where all the images are located
        @:param: filename - Name of the embeddings file to be used while saving
        @:param: save_dir - Path where the generated embeddings file will be saved
        @:param: allign - Whether to allign images before extracting embeddings
        @:param: resize - Wheter to resize images before extracting embeddings
        @:param: save - Whether to save extracted embeddings to a file  
    
    Returns:
          @:returns: A tuple of (np array embeddings, labels, image_paths)
    '''
    def generate_only_emebddings_at_path(self, image_dir, filename, save_dir="../Embeddings", allign=False, resize=False,save=True):

        dataset_embeddings, \
        dataset_boxes, \
        dataset_imagepaths = self.fe.get_only_embeddings_at_path(image_path=image_dir, allign=allign,
                                                            resize=resize, save_path=save_dir, save_to_file=save,
                                                            filename=filename)

        return dataset_embeddings, dataset_boxes, dataset_imagepaths

###################################################################################################################################
#
# This functions computes and generates dataset in a batch
#
###################################################################################################################################
    '''
    Params:
        @:param: image_path - Directory of image dataset
        @:param: batch_size - Number of images to be batched together
        @:param: resizeX , resizeY - Batching requires all images to be of same size. 
                                     This will be size images will be resized to
    
    Returns:
        @:returns: A tuple of (embeddings, labels, image_paths)
    '''
    def generate_dataset_batch(self, image_path, batch_size=8, resizeX=400, resizeY=400):
        dataset_embeddings, \
        dataset_labels, \
        dataset_imagepaths = self.fe.get_embeddings_batch(image_path=image_path, batch_size=batch_size,
                                                               resizeX=resizeX, resizeeY=resizeY)

        return dataset_embeddings, dataset_labels, dataset_imagepaths

###################################################################################################################################
#
# This function loads an already generated embedding file from the given load path
#
###################################################################################################################################
    '''
    Params:
        @:param: filename - Name of the embeddings/ feature file to load
        @:param: load_dir - Path to the embeddings directory 
        
    Returns:
        @:returns: A tuple of (np array of embeddings, labels, image_paths)
    '''
    def load_dataset(self, filename, load_dir):
        (dataset_embeddings, dataset_labels, dataset_imagepaths) = utilities.load_embeddings(
            load_path=load_dir,
            embed_filename=filename)

        print("Total Embeddings {}".format(np.array(dataset_embeddings).shape))
        print("Total Labels {}".format(np.array(dataset_labels).shape))
        print("Total Images {}".format(np.array(dataset_imagepaths).shape))

        return dataset_embeddings, dataset_labels, dataset_imagepaths

###################################################################################################################################
#
# This function loads an already generated only embeddings file from the given load path
#
###################################################################################################################################
    '''
    Params:
        @:param: filename - Name of the embeddings/ feature file to load
        @:param: load_dir - Path to the embeddings directory 

    Returns:
        @:returns: A tuple of (np array of embeddings, boxes, image_paths)
    '''

    def load_only_embeddings(self, filename, load_dir):
        (dataset_embeddings, dataset_boxes, dataset_imagepaths) = utilities.load_only_embeddings(load_path=load_dir, embed_filename=filename)

        print("Total Embeddings {}".format(np.array(dataset_embeddings).shape))
        print("Total Labels {}".format(np.array(dataset_boxes).shape))
        print("Total Images {}".format(np.array(dataset_imagepaths).shape))

        return dataset_embeddings, dataset_boxes, dataset_imagepaths

####################################################################################################################################
#
# Construct the argument parser and parse the arguments
#
###################################################################################################################################
def parse_args():


    ap = argparse.ArgumentParser()

    ap.add_argument("-id",
                    "--image_dataset_dir",
                    required=True,
                    help="Path to the image dataset")


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

    ap.add_argument("-eds",
                    "--embeddings_save_dir",
                    required=False,
                    help="Path where to save the generated embeddings",
                    default="../Embeddings\\")

    ap.add_argument("-fd",
                    "--face_detection_method",
                    choices=["CNN", "HOG"],
                    required=False,
                    help="type of face detection using HOG or deep learning CNN",
                    default="CNN")

    ap.add_argument("-fl",
                    "--face_landmarks_method",
                    choices=["68", "5"],
                    required=False,
                    help="Whether to use a 68-point or 5-point based landmark detection model",
                    default="68")

    ap.add_argument("-m",
                    "--mode",
                    choices=["load", "save"],
                    required=False,
                    help="Whether to load already existing embeddings or generate and save new ones",
                    default="load")

    ap.add_argument("-gpu",
                    "--use_gpu",
                    required=False,
                    default=True,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use GPU for computation")

    ap.add_argument("-a",
                    "--allign",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to allign images before computation")


    ap.add_argument("-mp",
                    "--multi_proc",
                    required=False,
                    default=True,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use multiprocessing")

    ap.add_argument("-b",
                    "--use_batch",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use batching")

    ap.add_argument("-bs",
                    "--batch_size",
                    required=False,
                    type=int,
                    default=8,
                    help="The number images in a batch if using batching")

    return vars(ap.parse_args())


if __name__ == '__main__':

    # Get the arguements
    args = parse_args()

    # Save the arguements
    ip = args["image_dataset_dir"]
    mode = args["mode"]
    gpu = args["use_gpu"]
    flm = args["face_landmarks_method"]
    fdm = args["face_detection_method"]
    embed_svdir = args["embeddings_save_dir"]
    embed_ldir = args["embeddings_load_dir"]
    filename = args["embed_filename"]
    use_batch = args["use_batch"]
    batch_size = args["batch_size"]
    allign = args["allign"]

    filename = "embeddings_ethnicity1"
    mode = "save"

    gen = GenerateDataset(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu)

    # Load embeddings
    if mode == "load":
        gen.load_dataset(filename=filename, load_dir=embed_ldir)

    # Else generate new ones
    else:
        # Use batching
        if use_batch:
            gen.generate_dataset_batch(image_path=ip, batch_size=batch_size)
        # Without batching
        else:
            gen.generate_dataset(image_dir =ip, filename=filename, save_dir=embed_svdir, allign=allign, resize=False, save=True)



