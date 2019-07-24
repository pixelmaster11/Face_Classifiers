
from imutils.face_utils import shape_to_np, FACIAL_LANDMARKS_IDXS
import numpy as np
import cv2



class FaceAlligner:

    def __init__(self, predictor, output_left_eye_coord=(0.35, 0.35), output_width=256, output_height=None):

        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.output_left_eye_coord = output_left_eye_coord
        self.output_width = output_width
        self.output_height = output_height

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.output_height is None:
            self.output_height = self.output_width

    def align_face(self, image, rect):

        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(image, rect)
        shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye_coords = shape[lStart:lEnd]
        right_eye_coords = shape[rStart:rEnd]

        # Get the center for each eye
        left_eye_center = left_eye_coords.mean(axis=0).astype("int")
        right_eye_center = right_eye_coords.mean(axis=0).astype("int")

        # compute the angle between the eyes tan^-1(dy / dx)
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]

        # Get the angle between the two eye centers
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        final_right_eye_X = 1.0 - self.output_left_eye_coord[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        final_distance = (final_right_eye_X - self.output_left_eye_coord[0])
        final_distance *= self.output_width
        scale = final_distance / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.output_width * 0.5
        tY = self.output_height * self.output_left_eye_coord[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        (w, h) = (self.output_width, self.output_height)

        output = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output