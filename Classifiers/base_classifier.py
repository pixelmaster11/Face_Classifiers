from abc import ABC, abstractmethod
import argparse
from Helper import utilities

'''
This is the base class for all types of ML classifier models
'''

class BaseClassifier(ABC):

    def __init__(self):
        self.args = self.parse_args()
        super().__init__()

    @abstractmethod
    def find_best_model_random(self, feat_train, lab_train):
        pass

    @abstractmethod
    def find_best_model_grid(self, feat_train, lab_train, ):
        pass

    ###################################################################################################################################
    #
    # Construct the argument parser and parse the arguments
    #
    ###################################################################################################################################
    def parse_args(self):
        ap = argparse.ArgumentParser()

        ap.add_argument("-s",
                        "--save_model",
                        required=False,
                        default=True,
                        type=utilities.str2bool,
                        nargs='?',
                        help="Whether to use save the trained model")

        ap.add_argument("-sd",
                        "--model_save_dir",
                        required=False,
                        help="Path where to save the trained model file",
                        default="../MLModels\\")


        ap.add_argument("-mf",
                        "--model_filename",
                        required=False,
                        help="Name of model file to be saved",
                        default="ml_model")


        ap.add_argument("-f",
                        "--embed_filename",
                        required=True,
                        help="Name of saved embedding file")

        ap.add_argument("-ld",
                        "--embed_load_dir",
                        required=False,
                        help="Path from where to load the embeddings",
                        default="../Embeddings\\")





        return vars(ap.parse_args())