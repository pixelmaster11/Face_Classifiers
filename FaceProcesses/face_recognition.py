
from Helper import similarity_metrics as sm
from Helper import utilities
from Helper.generate_dataset import GenerateDataset
import numpy as np
import cv2
import time
import argparse
import imutils
from imutils.video import VideoStream
from collections import Counter


'''
This class handles all face recognition stuff. 
Recognize faces from webcam, videofile or image directory
'''

class FaceRecognition:

###################################################################################################################################
    '''
    Params:
        @:param: face_detetion_model - Type of face detection to use CNN or HOG
        @:param: face_landmark_model - Type of face landmark detetction to use 68-point or 5-point based
        @:param: use_gpu - Whether to use GPU during computation
    '''
###################################################################################################################################

    def __init__(self):

        self.dataset_embeddings = None
        self.dataset_labels = None
        self.dataset_imagepaths = None


###################################################################################################################################
#
# This function is used for face recognition for an image directory
#
###################################################################################################################################
    '''
        @:param: target_path - Directory of all images for which to perform recognition
        @:param: distance_threshold - Loose / Strict allowance of false matches
        @:param: metric - Metric used to calculate distance between embeddings. Choices are euclidean, euclidean_numpy, cosine
        @:param: allign - Whether to allign images for improved accuracy
        @:param: reesize - Whether to resize images for memory allocs
    '''
    def recognize(self, target_path, distance_threshold = 0.5, metric = "euclidean", allign = False, resize = False,
                  save = False, save_dir = "../Embeddings", filename = "embeddings_fr_test"):


        #TODO: Do not allign when using batched images
        #if use_batch:
        #    allign = False

        print("\nRecognizing faces at path %s" % target_path)

        target_embeddings, boxes, target_images = gen.generate_only_emebddings_at_path(image_dir=target_path,
                                                                               allign=allign, resize=resize, save=save,
                                                                               save_dir=save_dir, filename=filename)
            
        print(np.array(target_embeddings).shape)
        print(np.array(target_images).shape)
        print(np.array(boxes).shape)

        self._calculate_matches(target_embeddings=target_embeddings, dets = boxes,
                                threshold = distance_threshold, target_images=target_images, metric=metric)



###################################################################################################################################
#
# This function performs the matching and returns the result of face recognition
#
###################################################################################################################################
    '''
        @:param: target_embeddings - List of embeddings to match with the dataset embeddings
        @:param: target_images  - List of images for which to perform face recognition
    '''
    def _calculate_matches(self, target_embeddings, target_images, dets, threshold=0.5, metric="euclidean"):
        matches = []
        names = []

        print(np.array(self.dataset_embeddings).shape)

        # TODO: Different shape returned when used batching
        # if use_batch:
        #    axis = 0
        # else:
        #    axis = 1

        axis = 1

        for num, target_embedding in enumerate(target_embeddings):
            for i, d in enumerate(self.dataset_embeddings):

                # convert dlib embedding vectors to numpy arrays
                d = np.array(d)
                target_embedding = np.array(target_embedding)

                # Incase if some of the embeddings is Null skip it
                if d.ndim == 0:
                    continue

                # Distance calculation
                if metric == "euclidean":
                    target_distance = sm.euclidean_distance(np.array(d, dtype=np.float32).reshape(-1, 1),
                                                            np.array(target_embedding, dtype=np.float32).reshape(-1, 1))

                elif metric == "euclidean_numpy":
                    target_distance = sm.euclidean_distance_numpy(x=d, y=target_embedding, axis=axis)

                elif metric == "cosine":

                    target_distance = sm.cosine_distance(x=np.array(d, dtype=np.float32).reshape(-1, 1),
                                                         y=np.array(target_embedding, dtype=np.float32).reshape(-1, 1))

                else:
                    print("Please provide metrics as euclidean, euclidean_numpy, cosine")
                    exit()

                if target_distance < threshold:
                    matches.append(self.dataset_imagepaths[i])
                    names.append(self.dataset_labels[i])

            '''for i, ip in enumerate(matches):
                m = cv2.imread(ip)
                m = cv2.resize(m, (320, 320))
                cv2.putText(m, str(i), (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Match", m)
                cv2.waitKey(0)'''

            if fdm == "CNN":
                box = dets[num].rect
            else:
                box = dets[num]

            target_image = cv2.imread(target_images[num])
            r = target_image.shape[1] / float(target_image.shape[1])

            x = int(box.left() * r)
            y = int(box.top() * r)
            w = int(box.right() * r) - x
            h =  int(box.bottom() * r) - y



            # Scale large images when using GPU
            if (target_image.shape[1] > 1000 or target_image.shape[0] > 1000) and gpu:
                resize = True
            else:
                resize = False

            if resize:
                scale_percent = 50  # percent of original size
                width = int(target_image.shape[1] * scale_percent / 100)
                height = int(target_image.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                target_image = cv2.resize(target_image, dim, interpolation=cv2.INTER_AREA)

            cv2.rectangle(target_image, (x, y), (x + w, y + h), (0,255,0), 3)

            if len(names) > 0:
                print(Counter(names))
                most_common, num_most_common = Counter(names).most_common(1)[0]

                if num_most_common > 2:

                    cv2.putText(target_image, str(most_common), (box.left() - 10, box.top() - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                else:
                    cv2.putText(target_image, "Unknown", (box.left() - 10, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255), 2)

            else:
                cv2.putText(target_image, "Unknown", (box.left() - 10, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 2)

            cv2.imshow("Match", target_image)
            cv2.waitKey(0)

            matches.clear()
            names.clear()

###################################################################################################################################
#
# This function is used for face recognition through webcam feed or a video file
#
###################################################################################################################################
    '''
        @:param: use_video - Whether to use a video file as feed instead of webcam
        @:param: display - Whether to display the stream on screen
        @:param: write_output - Whether to write the detected faces from the stream to a file
        @:param: video_input_path - Path for the input video
        @param: video_output_path - Path where the written output video will be saved at 
    '''
    def recognize_web_cam(self, use_video = False, display = True, write_output = False,
                          video_input_path = "../Videos\\InputVid.mp4", video_output_path = "../Videos\\Outputvid.avi"):

        # initialize the video stream and pointer to output video file, then
        # allow the camera sensor to warm up
        print("Starting video stream...")

        if use_video:
            vs = cv2.VideoCapture(video_input_path)
        else:
            vs = VideoStream(src=0, framerate=60).start()

        writer = None
        time.sleep(2.0)

        target_embeddings = []
        names = []
        skip = False

        # loop over frames from the video file stream
        while True:

            if not use_video:
                # grab the frame from the threaded video stream
                rgb = vs.read()

            else:
                grabbed, rgb = vs.read()

                if not grabbed:
                    break

            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            #rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            #rgb = cv2.resize(rgb, (400, 300))
            rgb = imutils.resize(rgb, width=400)
            #r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face

            dets = gen.fe.fd.detect_face(image=rgb)

            embeds = gen.fe.get_embeddings(image=rgb, dets=dets, draw=True)

            if embeds is None:
                skip = True

            if not skip:
                # For multiple faces in single image
                for e in embeds:

                    if e is None:
                        skip = True
                    elif len(e) == 0:
                        skip = True

                    target_embeddings.append(e)


            if not skip:
                for num, target_embedding in enumerate(target_embeddings):

                    if fdm == "CNN":
                        box = dets[num].rect
                    else:
                        box = dets[num]

                    for i, d in enumerate(self.dataset_embeddings):

                        d = np.array(d)
                        target_embedding = np.array(target_embedding)

                        target_distance = sm.euclidean_distance(x=d, y=target_embedding)

                        if target_distance < 0.55:
                            names.append(self.dataset_labels[i])

                        if len(names) > 0:
                            print(Counter(names))
                            most_common, num_most_common = Counter(names).most_common(1)[0]

                            cv2.putText(rgb, str(most_common), (box.left() + 15, box.bottom() + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                            writer = cv2.VideoWriter(video_output_path, fourcc, 20,(rgb.shape[1], rgb.shape[0]), True)
                        #else:
                        #    cv2.putText(rgb, "Unknown", (box.left() - 15, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

                target_embeddings.clear()
                names.clear()

            # if writer is not None, write frame with recognized faces
            if writer is not None:
                writer.write(rgb)

            if display:
                cv2.imshow("Frame", rgb)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            skip = False

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        # check if the vid writer point needs to be released
        if writer is not None:
            writer.release()

###################################################################################################################################
#
# Construct the argument parser and parse the arguments
#
###################################################################################################################################
def parse_args():


    ap = argparse.ArgumentParser()

    ap.add_argument("-id",
                    "--image_dir",
                    required=False,
                    help="Path for the image dataset",
                    default="../Images\\Face_Recognition\\")


    ap.add_argument("-tid",
                    "--test_image_dir",
                    required=False,
                    help="Path for the test images",
                    default="../Images\\Test_Images\\Face_Recognition\\")


    ap.add_argument("-eds",
                    "--embeddings_save_dir",
                    required=False,
                    help="Path where to save the test embeddings file",
                    default="../Embeddings\\")


    ap.add_argument("-edl",
                    "--embeddings_load_dir",
                    required=False,
                    help="Path where to load the saved embeddings file",
                    default="../Embeddings\\")

    ap.add_argument("-ef",
                    "--embed_filename",
                    required=False,
                    help="Name of saved embeddings file",
                    default="embeddings.pkl")


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
    ip = args["image_dir"]
    test_ip = args["test_image_dir"]
    gpu = args["use_gpu"]
    flm = args["face_landmarks_method"]
    fdm = args["face_detection_method"]
    embed_svdir = args["embeddings_save_dir"]
    embed_ldir = args["embeddings_load_dir"]
    use_cam = args["use_cam"]
    use_vid = args["use_vid"]

    filename = args["embed_filename"]

    mode = "load"
    embed_ldir = "../Embeddings\\"
    filename = "embeddings_fr.pkl"
    gpu = True


    gen = GenerateDataset(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu)
    fr = FaceRecognition()

    # Load dataset from given path
    fr.dataset_embeddings, fr.dataset_labels, fr.dataset_imagepaths = gen.load_dataset(filename=filename ,load_dir=embed_ldir)

    # Perform face recognition on test image directory
    if not use_cam and not use_vid:
        fr.recognize(target_path=test_ip, distance_threshold=0.6, metric="euclidean", allign=False, resize=False)

    # Face recognition on webcam or video input feed
    else:
        fr.recognize_web_cam(use_video=use_vid, display=True, write_output=True)




