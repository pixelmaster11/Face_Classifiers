
import dlib
import cv2
import imutils
from imutils import face_utils, paths
import numpy as np
from face_allignment import FaceAlligner

'''
This class handles all functions related to face processing.
Functions like detecting faces, alligning them, slicing them is handled here

'''

class FaceDetection:

    # Load the face detection and recognition models
    def __init__(self, face_detection_model = "HOG", face_landmark_model = "68"):


        # Load and set appropriate 68point and 5 point face landmark detection models
        self.shape_68_face_landmarks = dlib.shape_predictor("Dlib\shape_predictor_68_face_landmarks.dat")
        self.shape_5_face_landmarks = dlib.shape_predictor("Dlib\shape_predictor_68_face_landmarks.dat")
        print("Using {} points face landmark detection model".format(face_landmark_model))


        if face_landmark_model == "68":
            self.sp = self.shape_68_face_landmarks
        elif face_landmark_model == "5":
            self.sp = self.shape_5_face_landmarks
        else:
            print("Please provide 68 or 5 as strings for face landmark detection models")
            exit()



        # Load and set appropriate HOG and CNN based face detection model; CNN is extremely slower and needs a powerful GPU to run quickly
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1("Dlib\mmod_human_face_detector.dat")
        self.faces = dlib.full_object_detections()




        print("Using {} face detection model".format(face_detection_model))
        if face_detection_model == "HOG":
            self.detector = self.hog_face_detector
        elif face_detection_model == "CNN":
            self.detector = self.cnn_face_detector
        else:
            print("Please provide HOG or CNN as string parameters for face detection model")
            exit()

###########################################################################

    # Load the image from given image path
    '''
    Params:
        image_path - Path from where to load the image
        load_rgb_using_dlib - Load an RGB image using dlib library, otherwise image will be loaded using opencv
        resize_using_opencv - Whether to resize using opencv or imutils resize
        resize - Whether to resize image or not
        resize_width - Final width the given image will be resized to
    
    Returns:
        Returns the loaded image:
            BGR format using opencv, RGB using dlib
    
    '''
    def load_image(self, image_path, load_rgb_using_dlib = False, resize_using_opencv = True, resize = False, resize_width = 500):


        if image_path is None:

            print("Please provide image or image path to read the image data")
            exit()

        if load_rgb_using_dlib:
            image = dlib.load_rgb_image(image_path)
        else:
            # load the input image, resize it
            image = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)

        if resize:
            if resize_using_opencv:
                image = cv2.resize(image, (resize_width, resize_width))
            else:
                image = imutils.resize(image, width=resize_width)

        return image

###########################################################################

    # Detects a face from the given image
    '''
    Params:
        image - input image to detect the face
        upsample - Whether to expand the image to improve detection of smaller images
    Returns: 
        If using a CNN detector:
            A dlib MMOD (max-margin object-detection) rectangle object
            You can access its rectangles as [det.rect for det in dets] 
        If using a HOG detector:
            Returns the detections of type dlib rectangles which form the bounding boxes of detection
            You can directly access the rectangles as [for rect in dets]
    '''
    def detect_face(self, image, upsample = 1):

        print("\nDetecting face..")
        dets = self.detector(image, upsample)

        print("Detection Type %s" %type(dets))
        print("Number of faces detected: {}".format(len(dets)))
        return dets

###########################################################################

    # Detects faces from the given image list in batches
    '''
    Params:
        image_list = A list of images from which faces are to be detected
        upsample = Whether to upsample images before detecting for improving detection of smaller size images
        batch_size = Batch size to be inferred together
    
    Returns:
        A dlib MMOD (max-margin object-detection) rectangle object
        You can access its rectangles as det.rect for det in dets 
    '''
    def detect_face_batch(self, image_list, upsample = 1, batch_size = 128, resize = False, sizeX= 320, sizeY=320):

        print("\nDetecting faces in a batch...")

        # Force Load the cnn detector if chosen otherwise
        self.detector = self.cnn_face_detector
        dets = None

        try:
            # Get the detections in batches
            dets = self.detector([image for image in image_list], upsample, batch_size)

        # Check if all images have same sizes, if not then resize the images
        except RuntimeError as re:
            if hasattr(re, 'message'):
                print(re.message)
            else:
                print(re)

            # Resize all images to the same size
            print("Resizing all images to {} x {}".format(sizeX, sizeY))
            resized_images = []
            [resized_images.append(cv2.resize(image, (sizeX, sizeY)))  for image in image_list]
            image_list = resized_images
            del resized_images

            # Get the detections for all images in batches
            dets = self.detector([image for image in image_list], upsample, batch_size)

        # Return detections
        finally:
            print("Detection type %s" %type(dets))
            return dets

###########################################################################

    '''
    Params:
        image = Input image
        dets = Detected rectangles using Hog or CNN face detector
    
    Returns:
        List of face landmarks locations
    '''
    def detect_face_landmarks(self, image, dets):

        print("\nDetecting facial landmarks..")

        shapes = []

        # If Hog based detection
        if self.detector == self.hog_face_detector:

            # Loop over the face detections
            for rect in dets:

                # Determine the facial landmarks
                shape = self.sp(image, rect)
                shapes.append(shape)

        # If CNN based detection
        else:
            # Loop over the face detections
            for d in dets:
                # Determine the facial landmarks
                shape = self.sp(image, d.rect)
                shapes.append(shape)

        #for shape in shapes:
        #    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                              shape.part(1)))

        print("Landmarks successfully detected")
        return shapes

###########################################################################

    # Function alligns an image into a 150 x 150 image using dlib library
    '''
    Params:
        image - Input image to be alligned
        dets - Detection rectangles 
        padding - Extra padding to be added to the output alligned image
        size - Image to be resized to this value 
    
    Returns:
        A list of alligned face images
    '''
    # Get alligned face crops using dlib library
    def get_face_chips(self, image, dets, padding = 0.25, size = 320):

        print("\nAlligning face using dlib..")

        # If HOG based face detection
        if self.detector == self.hog_face_detector:

            for detection in dets:
                self.faces.append(self.sp(image, detection))

        # If CNN based face detection
        else:
            for detection in dets:
                self.faces.append(self.sp(image, detection.rect))

        # Get the alligned face images
        images = dlib.get_face_chips(image, self.faces, size= size, padding = padding)

        print("Image successfully alligned and resized to size %s" %size)

        return images

###########################################################################

    # Get Alligned faces using a custom face alligner based on face landmarks (For learning refer Pyimagesearch)
    '''
    Params:
        image - Loaded input image
        output_width - Final width of alligned image
        dets - Detection rects 
    
    Returns:
        A list of alligned images
    '''
    def get_alligned_face(self, image, output_width=256, dets = None):

        print("\nAlligning face with custom Affine Transform Alligner..")

        # Initialize Face Alligner
        alligned_faces = []
        fa = FaceAlligner(self.sp, output_width=output_width)

        if image is None:
            print("Please provide an image to allign")
            exit()


        if dets is None:
            dets = self.detect_face(image=image)


        if dets == 0:
            print("No face detected")
            return None



        # Loop over the face detections
        for (i, d) in enumerate(dets):

            # Determine the facial landmarks
            # If HOG based fece detection
            if self.detector == self.hog_face_detector:
                rect = d
            # Else CNN based face detection
            else:
                rect = d.rect

            alligned_faces.append(fa.align_face(image, rect))

        print("Image successfully alligned and resized to size %s" % output_width)

        return alligned_faces

###########################################################################
    # Get face slices of each landmark detected
    '''
    Params:
        image - Input image to slice
        dets - Detection rects
        slice_width - Width of each slice to be resized to
    
    Returns:
        A list of slices with following indices (For 68 point landmarks model):-
        ("mouth", (48, 68)),
	    ("right_eyebrow", (17, 22)),
	    ("left_eyebrow", (22, 27)),
	    ("right_eye", (36, 42)),
	    ("left_eye", (42, 48)),
	    ("nose", (27, 36)),
	    ("jaw", (0, 17))
	    
	    A list of slices with following indices (For 5 point landmarks model):-
	    ("right_eye", (2, 3)),
	    ("left_eye", (0, 1)),
	    ("nose", (4))
    '''
    def get_face_slices(self, image, dets = None, slice_width = 256):


        ROI_list = []

        # Set the landmark locations based on 68or 5 point landmark detection model used
        if self.sp == self.shape_68_face_landmarks:
            face_landmarks_list = face_utils.FACIAL_LANDMARKS_68_IDXS
        else:
            face_landmarks_list = face_utils.FACIAL_LANDMARKS_5_IDXS

        if image is None:
            print("Please provide image to slice")
            exit()

        if dets is None:
            # Detect faces in the image
            dets = self.detect_face(image, 1)

        if dets == 0:
            print("No face detected")
            return None



        # Loop over the face detections to  determine the facial landmarks
        for (s, d) in enumerate(dets):

           # Check for detection method used HOG or CNN
            if self.detector == self.hog_face_detector:
                rect = d
            else:
                rect = d.rect

            shape = self.sp(image, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the face parts individually
            for (name, (i, j)) in face_landmarks_list.items():

                # Extract the ROI of the face region as a separate image

                #print("Name {}, co-ordinates {}".format(name, (i, j)))
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=slice_width, inter=cv2.INTER_CUBIC)

                ROI_list.append((roi, name))


        return ROI_list

###########################################################################

    # Draw face landmarks on the image
    '''
    Params:
        image - Given input image on which to draw face landmarks
        dets - Detection rectangles of the given input image
        shapes - Located landmarks shapes 
        return_drawn_landmarks - Whether to return the image with drawn landmarks or display using this function without returning anything
        draw_type - Whether to draw points or lines to depict a landmark
    '''
    def draw_face_landmarks(self, image, dets = None, shapes = [], return_drawn_landmarks = False, draw_type = "line"):

        print("\nDrawing face landmarks..\n")
        win = dlib.image_window()
        win.set_image(image)

        if self.sp == self.shape_68_face_landmarks:
            face_landmarks_list = face_utils.FACIAL_LANDMARKS_68_IDXS
        else:
            face_landmarks_list = face_utils.FACIAL_LANDMARKS_5_IDXS

        if image is None:
            print("Please provide an image")
            exit()

        if dets is None:
            dets = self.detect_face(image=image)

        if len(shapes) == 0:
            shapes = self.detect_face_landmarks(image=image, dets=dets)


        for shape in shapes:
            win.add_overlay(shape, dlib.rgb_pixel(0,255,0))

        # Draw landmarks over the image using opencv line or circle to return the drawn image
        if return_drawn_landmarks:
            for shape in shapes:
                shape = face_utils.shape_to_np(shape)

                # Loop over the face parts individually
                for (name, (i, j)) in face_landmarks_list.items():

            # Loop over the subset of facial landmarks, drawing the
            # specific face part

                    px = None
                    py = None

                    for (x, y) in shape[i:j]:

                        if draw_type == "line":
                            if px is None and py is None:
                                px,py = x,y
                            cv2.line(image, (px, py), (x, y), (0, 255, 0), 2)
                            px, py = x, y

                        else:
                            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


            return image

        else:
            dlib.hit_enter_to_continue()
            return image



if __name__ == "__main__":

    # Testing all functionality
    fd = FaceDetection(face_detection_model="HOG")

    images = []

    for image_path in paths.list_images("D:\Tuts\DataScience\Python\Datasets\FGNET\Age_Test\Old"):
        image = fd.load_image(image_path=image_path)
        images.append(image)

    # Test with only last image from list which has 3 faces
    dets = fd.detect_face(images[-1])

    shapes = fd.detect_face_landmarks(images[-1], dets)

    image = fd.draw_face_landmarks(image=images[-1], dets=dets, shapes=shapes, return_drawn_landmarks=False)

    alligned_chips = fd.get_alligned_face(image=images[-1], dets=dets, output_width=500)


    for ac in alligned_chips:
        cv2.imshow("Alligned_Face", ac)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    slices = fd.get_face_slices(image=images[-1], dets=dets)

    for slice, name in slices:
        cv2.imshow(str(name).upper(), slice)
        cv2.waitKey(0)
        cv2.destroyWindow(str(name).upper())


