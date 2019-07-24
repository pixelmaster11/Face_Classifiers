
from FaceProcesses.face_encodings import Face_Encoding
from Helper import utilities
import numpy as np
import argparse

class GenerateDataset():

    def __init__(self, face_detection_model="CNN", face_landmark_model="68", use_gpu=True):

        self.fe = Face_Encoding(face_detection_model=face_detection_model, face_landmark_model=face_landmark_model,
                                use_gpu=use_gpu)
        self.dataset_embeddings = None
        self.dataset_labels = None
        self.dataset_imagepaths = None


    def generate_dataset(self, image_dir, filename, save_dir="../Embeddings", allign=False, resize=False,save=True):
        self.dataset_embeddings, \
        self.dataset_labels, \
        self.dataset_imagepaths = self.fe.get_embeddings_at_path(image_path=image_dir, allign=allign,
                                                                 resize=resize, save_path=save_dir, save_to_file=save,
                                                                 filename=filename)

    def generate_dataset_batch(self, image_path, batch_size=8, resizeX=400, resizeY=400):
        self.dataset_embeddings, \
        self.dataset_labels, \
        self.dataset_imagepaths = self.fe.get_embeddings_batch(image_path=image_path, batch_size=batch_size,
                                                               resizeX=resizeX, resizeeY=resizeY)

    # This function loads an already generated embedding file from the given load path
    def load_dataset(self, filename, load_dir):
        (self.dataset_embeddings, self.dataset_labels, self.dataset_imagepaths) = utilities.load_embeddings(
            load_path=load_dir,
            embed_filename=filename)

        print("Total Embeddings {}".format(np.array(self.dataset_embeddings).shape))
        print("Total Labels {}".format(np.array(self.dataset_labels).shape))
        print("Total Images {}".format(np.array(self.dataset_imagepaths).shape))



# Construct the argument parser and parse the arguments
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
                    help="Name of embeddings file that will be saved",
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
    filename = args["embed_filename"]
    embed_ldir = args["embeddings_load_dir"]
    use_batch = args["use_batch"]
    batch_size = args["batch_size"]

    filename = "embeddings_gender"
    mode = "save"

    gen = GenerateDataset(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu)

    # Load embeddings else generate new ones
    if mode == "load":
        gen.load_dataset(filename=filename, load_dir=embed_ldir)
    else:
        if use_batch:
            gen.generate_dataset_batch(image_path=ip, batch_size=batch_size)
        else:
            gen.generate_dataset(image_dir =ip, filename=filename, save_dir=embed_svdir, allign=False, resize=False, save=True)



