
import dlib
import cv2
import imutils
from imutils import face_utils, paths
import numpy as np
from face_allignment import FaceAlligner


class FaceDetection:


    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor("Dlib\shape_predictor_68_face_landmarks.dat")
        self.facerec = dlib.face_recognition_model_v1("Dlib/dlib_face_recognition_resnet_model_v1.dat")
        self.faces = dlib.full_object_detections()

    def detect_face(self, image, upsample = 1):

        dets = self.detector(image, upsample)

        return dets, image

    def detect_face_landmarks(self, image, dets):

        shapes = []
        # loop over the face detections
        for (i, rect) in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.sp(image, rect)
            shapes.append(shape)

        return shapes

    def get_face_chips(self, image, dets):


        for detection in dets:
            self.faces.append(self.sp(image, detection))

        # Get the aligned face images
        # Optionally:
        # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
        images = dlib.get_face_chips(image, self.faces, size=320)

        return images

    def get_face_slices(self, image_path, debug = False):


        ROI_list = []
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_path)
        #cv2.imshow("Slice", image)
        #cv2.waitKey(0)

        image = imutils.resize(image, width=500)
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(image, 1)



        if rects == 0:
            print("No face detected")
            return None

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = self.sp(image, rect)
            shape = face_utils.shape_to_np(shape)

            fa = FaceAlligner(fd.sp, output_width=256)

           # cv2.imshow("0", image)
           # cv2.imshow("1", fa.align_face(image, rect))
           # cv2.waitKey(0)

            # loop over the face parts individually
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                # clone the original image so we can draw on it, then
                # display the name of the face part on the image
                #clone = image.copy()
                #cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.7, (0, 0, 255), 2)

                # loop over the subset of facial landmarks, drawing the
                # specific face part
                #for (x, y) in shape[i:j]:
                    #cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)


                # extract the ROI of the face region as a separate image

                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

                roi = image[y:y + h, x:x + w]
                roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

                ROI_list.append(roi)

                if debug:
                    # show the particular face part
                    cv2.imshow("ROI", roi)
                    cv2.waitKey(0)


        return ROI_list




if __name__ == "__main__":

    fd = FaceDetection()

    for image_path in paths.list_images("D:\Tuts\DataScience\Python\Datasets\FGNET\Age_Test\Old"):


        slices = fd.get_face_slices(image_path)

        for slice in slices:
            cv2.imshow("Slice", slice)
            cv2.waitKey(0)
