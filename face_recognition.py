
import similarity_metrics as sm
import numpy as np
import cv2
import os
import time
import utilities
import argparse
import imutils
from imutils.video import VideoStream
from collections import Counter
from face_encodings import Face_Encoding


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
    def __init__(self, face_detection_model = "CNN", face_landmark_model = "68", use_gpu = True):

        self.fe = Face_Encoding(face_detection_model=face_detection_model, face_landmark_model=face_landmark_model, use_gpu=use_gpu)
        self.dataset_embeddings = None
        self.dataset_labels = None
        self.dataset_imagepaths = None

###################################################################################################################################

    # This function generates an embedding file for all the given images at the image_path
    '''
    @:param: image_path - Path for all the images of which to calculate and save embeddings
    @:param: allign - Whether to allign images or not
    @:param: resize - Whether to resize images
    @:param: save - Whether to save generated embeddings to file
    @:param: filename - Embbedings save filename
    '''
    def generate_dataset(self, image_path, allign = True, resize = False, save = True, filename = "embeddings_fr"):
        self.dataset_embeddings, \
        self.dataset_labels, \
        self.dataset_imagepaths = self.fe.get_embeddings_at_path(image_path=image_path, allign=allign,
                                                                 resize=resize, save_path=embed_svdir, save_to_file=save ,
                                                                 filename=filename)

    def generate_dataset_batch(self, image_path, batch_size = 8, resizeX = 400, resizeY = 400):

        self.dataset_embeddings, \
        self.dataset_labels, \
        self.dataset_imagepaths = self.fe.get_embeddings_batch(image_path=image_path, batch_size=batch_size,
                                                               resizeX=resizeX, resizeeY=resizeY)

    # This function loads an already generated embedding file from the given load path
    def load_dataset(self, filename):

        (self.dataset_embeddings, self.dataset_labels, self.dataset_imagepaths) = utilities.load_embeddings(load_path=embed_ldir,
                                                                                                          embed_filename=filename)

        print("Total Embeddings {}".format(np.array(self.dataset_embeddings).shape))
        print("Total Labels {}".format(np.array(self.dataset_labels).shape))
        print("Total Images {}".format(np.array(self.dataset_imagepaths).shape))

###################################################################################################################################

    # This function is used for face recognition for an image directory
    '''
        @:param: target_path - Directory of all images for which to perform recognition
        @:param: distance_threshold - Loose / Strict allowance of false matches
        @:param: metric - Metric used to calculate distance between embeddings. Choices are euclidean, euclidean_numpy, cosine
        @:param: allign - Whether to allign images for improved accuracy
        @:param: reesize - Whether to resize images for memory allocs
    '''
    def recognize(self, target_path, distance_threshold = 0.5, metric = "euclidean", allign = True, resize = False):

        target_embeddings = []
        target_images = []
        boxes = []

        # Do not allign when using batched images
        if use_batch:
            allign = False

        print("\nRecognizing faces at path %s" % target_path)

        #For all images in dir
        for ip in os.listdir(target_path):

            ip = os.path.join(target_path, ip)
            print("\n"+ip)
            image = cv2.imread(ip)
            t1 = time.time()

            if (image.shape[1] > 1000 or image.shape[0] > 1000) and gpu:
                resize = True
            else:
                resize = False

            # Resize target images
            if resize:
                scale_percent = 50  # percent of original size
                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)
                # resize image
                image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # Allign images
            if allign:
                dets = self.fe.fd.detect_face(image=image)
                alligned_images = self.fe.fd.get_alligned_face(image=image, dets=dets,output_width=256)

                # For each alligned image in case of multiple faces in a single image
                for image in alligned_images:

                    # Get the embeddings for the image
                    e = self.fe.get_embeddings(image=image)

                    if e is None:
                        continue
                    elif len(e) == 0:
                        continue

                    target_embeddings.append(e)
                    target_images.append(image)

            #If no allignment then directly compute the embeddings
            else:
                dets = self.fe.fd.detect_face(image=image)
                embeds = self.fe.get_embeddings(image, dets=dets)

                if embeds is None:
                    continue

                # For multiple faces in single image
                for e in embeds:

                    if e is None:
                        continue
                    elif len(e) == 0:
                        continue

                    target_embeddings.append(e)
                    target_images.append(image)

            t2 = time.time()

            for d in dets:
                boxes.append(d)
            
            print("Time taken to extract embedding: %f"  %(t2-t1))
            
        print(np.array(target_embeddings).shape)
        print(np.array(target_images).shape)
        
        self._calculate_matches(target_embeddings=target_embeddings, dets = boxes,
                                threshold = distance_threshold, target_images=target_images, metric=metric)

###################################################################################################################################

    # This function is used for face recognition through webcam feed or a video file
    '''
        @:param: use_video - Whether to use a video file as feed instead of webcam
        @:param: display - Whether to display the stream on screen
        @:param: write_output - Whether to write the detected faces from the stream to a file
        @:param: video_input_path - Path for the input video
        @param: video_output_path - Path where the written output video will be saved at 
    '''
    def recognize_web_cam(self, use_video = False, display = True, write_output = False,
                          video_input_path = "Videos\\InputVid.mp4", video_output_path = "Videos\\Outputvid.avi"):

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
            rgb = imutils.resize(rgb, width=750)
            #r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face

            dets = self.fe.fd.detect_face(image=rgb)

            embeds = self.fe.get_embeddings(image=rgb, dets=dets)

            if embeds is None:
                continue

            # For multiple faces in single image
            for e in embeds:

                if e is None:
                    continue
                elif len(e) == 0:
                    continue

                target_embeddings.append(e)



            for num, target_embedding in enumerate(target_embeddings):

                if fdm == "CNN":
                    box = dets[num].rect
                else:
                    box = dets[num]

                for i, d in enumerate(self.dataset_embeddings):

                    d = np.array(d)
                    target_embedding = np.array(target_embedding)

                    target_distance = sm.euclidean_distance_numpy(x=d, y=target_embedding, axis=1)

                    if target_distance < 0.5:
                        names.append(self.dataset_labels[i])

                    if len(names) > 0:
                        #print(Counter(names))
                        most_common, num_most_common = Counter(names).most_common(1)[0]

                        cv2.putText(rgb, str(most_common), (box.left() - 15, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        writer = cv2.VideoWriter(video_output_path, fourcc, 20,(rgb.shape[1], rgb.shape[0]), True)
                    #else:
                    #    cv2.putText(rgb, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

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

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        # check if the vid writer point needs to be released
        if writer is not None:
            writer.release()

###################################################################################################################################


    # This function performs the matching and returns the result of face recognition
    '''
        @:param: target_embeddings - List of embeddings to match with the dataset embeddings
        @:param: target_images  - List of images for which to perform face recognition
    '''
    def _calculate_matches(self, target_embeddings, target_images, dets, threshold = 0.5, metric = "euclidean"):
        
        matches = []
        names = []

        print(np.array(self.dataset_embeddings).shape)

        # Different shape returned when used batching
        if use_batch:
            axis = 0
        else:
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

            if len(names) > 0:
                print(Counter(names))
                most_common, num_most_common = Counter(names).most_common(1)[0]

                if num_most_common > 2:


                    cv2.putText(target_images[num], str(most_common), (box.left() - 10, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

                else:
                    cv2.putText(target_images[num], "Unknown", (box.left() - 10, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            else:
                cv2.putText(target_images[num], "Unknown", (box.left() - 10, box.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


            cv2.imshow("Match", target_images[num])
            cv2.waitKey(0)

            matches.clear()
            names.clear()

###################################################################################################################################

# Construct the argument parser and parse the arguments
def parse_args():


    ap = argparse.ArgumentParser()

    ap.add_argument("-id",
                    "--image_dir",
                    required=False,
                    help="Path for the image dataset",
                    default="Images\\Face_Recognition\\")


    ap.add_argument("-tid",
                    "--test_image_dir",
                    required=False,
                    help="Path for the test images",
                    default="Images\\Test_Images\\Face_Recognition\\")

    ap.add_argument("-edl",
                    "--embeddings_load_dir",
                    required=False,
                    help="Path where to load the saved embeddings file",
                    default="Embeddings\\")

    ap.add_argument("-ef",
                    "--embed_filename",
                    required=False,
                    help="Name of saved embeddings file",
                    default="embeddings.pkl")

    ap.add_argument("-eds",
                    "--embeddings_save_dir",
                    required=False,
                    help="Path where to save the generated embeddings",
                    default="Embeddings\\")

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
    ip = args["image_dir"]
    test_ip = args["test_image_dir"]
    mode = args["mode"]
    gpu = args["use_gpu"]
    flm = args["face_landmarks_method"]
    fdm = args["face_detection_method"]
    embed_svdir = args["embeddings_save_dir"]
    embed_ldir = args["embeddings_load_dir"]
    use_cam = args["use_cam"]
    use_vid = args["use_vid"]
    use_batch = args["use_batch"]
    batch_size = args["batch_size"]
    filename = args["embed_filename"]

    mode = "load"
    embed_ldir = "Embeddings\\"
    filename = "embeddings_fr.pkl"
    gpu = True


    fr = FaceRecognition(face_detection_model=fdm, face_landmark_model=flm, use_gpu=gpu)

    # Load embeddings else generate new ones
    if mode == "load":
        fr.load_dataset(filename=filename)
    else:
        if use_batch:
            fr.generate_dataset_batch(image_path=ip, batch_size=batch_size)
        else:
            fr.generate_dataset(image_path=ip, allign=True, resize=False)


    # Perform face recognition on image directory
    if not use_cam and not use_vid:
        fr.recognize(target_path=test_ip, distance_threshold=0.6, metric="euclidean", allign=False, resize=False)

    # Face recognition on webcam or video input feed
    else:
        fr.recognize_web_cam(use_video=use_vid, display=True, write_output=True)




