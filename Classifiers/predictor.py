
from Classifiers import ml_utils
from Helper import utilities
from Helper.generate_dataset import GenerateDataset
import argparse
import cv2

'''
This class is used only to show the predictions on the image using an already trained classifier model
'''


class Predictor():

    def __init__(self):
        self.model = None
        self.model_name = ""

###################################################################################################################################
#
# Loads the classifier model
#
###################################################################################################################################
    '''
    Params:
        @:param: filename - Name of ml model file to load
        @:param: load_dir - Directory from where the ml model will be loaded
    '''
    def load_model(self, filename, load_dir = "../MLModels/"):
        ml_model, ml_name, train_labels, train_features = ml_utils.load_ml_model(load_dir=load_dir, filename=filename)
        self.model = ml_model
        self.model_name = ml_name

###################################################################################################################################
#
# Predict
#
###################################################################################################################################
    '''
    Params:
        @:param: target_path - Path from where all images should be predicted
        @:param: allign - Whether to allign images before predicting    
    '''
    def predict(self, target_path, allign = False):

        target_embeddings, boxes, target_imagepaths = gen.generate_only_emebddings_at_path(image_dir=target_path, allign=allign, save=False)
        target_embeddings =  ml_utils.get_scaling(scaling_type="Norm").fit(target_embeddings).transform(target_embeddings)


        a_index = 0
        prev_ip = ""

        for i, ip in enumerate(target_imagepaths):
            image = cv2.imread(ip)
            target_feature = target_embeddings[i]
            target_det = boxes[i]

            r = image.shape[1] / float(image.shape[1])

            # Scale large images when using GPU
            if (image.shape[1] > 1000 or image.shape[0] > 1000) and gpu:
                resize = True
            else:
                resize = False

            if resize:
                scale_percent = 50  # percent of original size
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)

                # resize image
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


            prediction = self.model.predict(target_feature.reshape(1,-1))


            if fdm == "CNN":
                box = boxes[i].rect
            else:
                box = boxes[i]

            # For allign, get alligned images from path and detect their face again for proper bounding boxes
            if allign:
                alligned_images = gen.fe.fd.get_alligned_face(image=image)

                # If it is a new image at new path
                if ip != prev_ip:
                    a_index = 0
                    det = gen.fe.fd.detect_face(image=alligned_images[a_index])
                    prev_ip = ip

                # If there are multiple faces in single image, then increment index to get the next alligned image
                else:
                    a_index += 1
                    det = gen.fe.fd.detect_face(image=alligned_images[a_index])

                image = alligned_images[a_index]


                if fdm == "CNN":
                    box = det[0].rect
                else:
                    box = det[0]

            x = int(box.left() * r)
            y = int(box.top() * r)
            w = int(box.right() * r) - x
            h = int(box.bottom() * r) - y

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            cv2.putText(image, prediction[0], (box.left() + 10, box.top() + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow("Image",image)
            cv2.waitKey(0)

###################################################################################################################################
#
# Construct the argument parser and parse the arguments
#
###################################################################################################################################
def parse_args():


    ap = argparse.ArgumentParser()


    ap.add_argument("-tid",
                    "--test_image_dir",
                    required=False,
                    help="Path for the test images",
                    default="../Images\\Test_Images\\Gender_Recognition\\")


    ap.add_argument("-mdl",
                    "--model_load_dir",
                    required=False,
                    help="Path where to load the saved embeddings file",
                    default="../MLModels\\")

    ap.add_argument("-mf",
                    "--model_filename",
                    required=False,
                    help="Name of saved model file",
                    default="SVC_gender_recog.pkl")



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

    ap.add_argument("-a",
                    "--allign",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to allign images before computation")


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


    ap.add_argument("-wc",
                    "--use_cam",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use webcam feed for face recogntion")


    ap.add_argument("-vid",
                    "--use_vid",
                    required=False,
                    default=False,
                    type=utilities.str2bool,
                    nargs='?',
                    help="Whether to use a video for face recognition")


    return vars(ap.parse_args())



if __name__ == '__main__':
    # Get the arguements
    args = parse_args()

    # Save the arguements
    test_ip = args["test_image_dir"]
    gpu = args["use_gpu"]
    flm = args["face_landmarks_method"]
    fdm = args["face_detection_method"]
    allign = args["allign"]
    model_ldir = args["model_load_dir"]
    use_cam = args["use_cam"]
    use_vid = args["use_vid"]
    filename = args["model_filename"]
    allign = True

    # Load the ML Model to use for prediction
    gen = GenerateDataset(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu, verbose=1)
    predictor = Predictor()
    predictor.load_model(filename=filename, load_dir=model_ldir)

    # Predict using the model
    predictor.predict(target_path=test_ip, allign=allign)
