
import utilities
from face_encodings import Face_Encoding
import cv2
import numpy as np
import pickle
import similarity_metrics as sm
import os
from collections import Counter
import dlib.cuda as cuda
import dlib

class FaceRecognition:

    def __init__(self, face_detection_model = "CNN", face_landmark_model = "68"):
        self.fe = Face_Encoding(face_detection_model=face_detection_model, face_landmark_model=face_landmark_model)
        self.dataset_embeddings = None
        self.dataset_labels = None
        self.dataset_imagepaths = None

    def generate_dataset(self, image_path):
        self.dataset_embeddings, self.dataset_labels, self.dataset_imagepaths = utilities.get_embeddings_at_path(image_path=image_path, resize=False,
                                                                                                                save_to_file=True, fe = self.fe)

    def load_dataset(self, embeddings_filepath, labels_filepath = None, image_filepaths = None):

        # Loading Features
        with open(embeddings_filepath, "rb") as infile:
            (self.dataset_embeddings, self.dataset_labels, self.dataset_imagepaths) = pickle.load(infile)

        print("\nLoaded embeddings file from {}".format(embeddings_filepath))
        print("Total Embeddings {}".format(np.array(self.dataset_embeddings).shape))
        print("Total Labels {}".format(np.array(self.dataset_labels).shape))
        print("Total Images {}".format(np.array(self.dataset_imagepaths).shape))

    def recognize(self, target_path, distance_threshold = 0.5, metric = "euclidean", allign = True):

        target_embeddings = []
        target_images = []

        print("\nRecognizing faces..")

        for ip in os.listdir(target_path):
            ip = os.path.join(target_path, ip)
            print("\n"+ip)
            image = cv2.imread(ip)

            if allign:
                dets = self.fe.fd.detect_face(image=image)
                alligned_images = self.fe.fd.get_alligned_face(image=image, dets=dets,output_width=256)

                for image in alligned_images:
                    e = utilities.get_embeddings(fe=self.fe, image=image)

                    if e is None:
                        continue
                    elif len(e) == 0:
                        continue

                    target_embeddings.append(e)
                    target_images.append(image)

            else:
                embeds = utilities.get_embeddings(image, self.fe)

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

        print(np.array(target_embeddings).shape)
        print(np.array(target_images).shape)

        matches = []
        names = []

        for num, target_embedding in enumerate(target_embeddings):
            for i, d in enumerate(self.dataset_embeddings):


                #print(np.array(target_embedding).shape)
                #print(np.array(d).shape)

                d = np.array(d)
                target_embedding = np.array(target_embedding)

                #print(d.ndim)

                # Incase if some of the embeddings is Null skip it
                if d.ndim == 0:
                    continue

                #print(i)

                if metric == "euclidean":
                    target_distance = sm.euclidean_distance(np.array(d, dtype=np.float32).reshape(-1,1),
                                                            np.array(target_embedding, dtype=np.float32).reshape(-1, 1))

                elif metric == "euclidean_numpy":
                    target_distance = sm.euclidean_distance_numpy(x = d, y = target_embedding, axis=1)

                elif metric == "cosine":
                    distance_threshold /= 10
                    target_distance = sm.cosine_distance(x = np.array(d, dtype=np.float32).reshape(-1,1),
                                                        y = np.array(target_embedding, dtype=np.float32).reshape(-1, 1))

                else:
                    print("Please provide metrics as euclidean, euclidean_numpy, cosine")
                    exit()

                if target_distance < distance_threshold:
                    matches.append(self.dataset_imagepaths[i])
                    names.append(self.dataset_labels[i])

            '''for i, ip in enumerate(matches):
                m = cv2.imread(ip)
                m = cv2.resize(m, (320, 320))
                cv2.putText(m, str(i), (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Match", m)
                cv2.waitKey(0)'''

            #print(len(names))

            if len(names) > 0:
                print(Counter(names))
                most_common, num_most_common = Counter(names).most_common(1)[0]
                print(most_common)

                cv2.putText(target_images[num], str(most_common), (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Match", target_images[num])
                cv2.waitKey(0)

            else:
                cv2.putText(target_images[num], "Unknown", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                            2)
                cv2.imshow("Match", target_images[num])
                cv2.waitKey(0)

            matches.clear()
            names.clear()


    def display_matches(self):
        pass



if __name__ == '__main__':




    fr = FaceRecognition()
    ip = "Images\\Face_Recognition\\"
    ep = "Embeddings\\embeddings.pkl"
    target_image_path = "Images\\Test_Images\\Face_Recognition\\"

    print("Active GPUS %s" % str(cuda.get_num_devices()))

    print("Cuda %s" % str(dlib.DLIB_USE_CUDA))

    #fr.generate_dataset(image_path=ip)

    fr.load_dataset(embeddings_filepath=ep)


    fr.recognize(target_path=target_image_path, distance_threshold=0.5)




