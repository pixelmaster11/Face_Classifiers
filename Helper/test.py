
from Helper import utilities
import numpy as np
import cv2
from FaceProcesses.face_detection import FaceDetection


'''
This is just a test module to test or play around with features, etc
'''

if __name__ == '__main__':

    dataset_embeddings, dataset_labels, dataset_imagepaths = utilities.load_embeddings(
        embed_filename="embeddings_ethnicity.pkl")

    fd = FaceDetection(face_detection_model="CNN")

    sorted_indices = dataset_embeddings[: , 1].argsort()

    for i in sorted_indices:

        ip = dataset_imagepaths[i]

        #if not str(dataset_imagepaths[i]).__contains__(".."):
        #    ip = "../" + dataset_imagepaths[i]
        #else:
        #    ip = dataset_imagepaths[i]

        print(ip)
        image = cv2.imread(ip)

        if (image.shape[1] > 1000 or image.shape[0] > 1000):
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

        dets = fd.detect_face(image=image)
        images = fd.get_alligned_face(image=image, dets=dets)

        for image in images:
            dets = fd.detect_face(image=image)
            shapes = fd.detect_face_landmarks(image=image, dets=dets)
            image = fd.draw_face_landmarks(image=image,dets=dets,shapes=shapes, return_drawn_landmarks=True)
            #image = cv2.resize(image, (300,300))
            cv2.imshow("Sorted", image)
            cv2.waitKey(0)
