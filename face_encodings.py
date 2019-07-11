
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import dlib
from face_detection import FaceDetection


class Face_Encoding:

###############################################


    def __init__(self, face_detection_model = "HOG", face_landmark_model = "68"):

        self.fd_model = face_detection_model
        self.fl_model = face_landmark_model
        self.facerec = dlib.face_recognition_model_v1("Dlib/dlib_face_recognition_resnet_model_v1.dat")
        self.fd = FaceDetection(face_detection_model=face_detection_model , face_landmark_model=face_landmark_model)

    #Local binary pattern of image
    def get_local_binary_pattern(self, image, image_path, numPoints = 24, radius = 8, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns


        image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numPoints,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)


        # return the histogram of Local Binary Patterns
        return hist, lbp

###############################################

    #Get color histogram for all color channels
    def get_color_hist(self, image):
        hist_vector = np.array([])
        bins = 16

        for channel in range(image.shape[2]):
            channel_hist = np.histogram(image[:, :, channel],
                                        bins=bins, range=(0, 255))[0]
            hist_vector = np.hstack((hist_vector, channel_hist))

        #hist = cv2.normalize(hist_vector)
        return hist_vector

###############################################

    #Get spatial histogram of the image
    def get_spatial_hist(self, image):
        spatial_size = (16, 16)
        spatial_image = cv2.resize(image, spatial_size,
                                   interpolation=cv2.INTER_AREA)
        spatial_vector = spatial_image.ravel()
        return spatial_vector

###############################################

    #Get the hsv color channel histogram
    def get_hsv_hist(self, image):
        bins = 16
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])

        hist = cv2.normalize(hist)

        # return the flattened histogram as the feature vector
        td_hist = hist.flatten()

        return td_hist

###############################################

    #Get the Histogram of Oriented Gradients for the given  image
    def get_hog(self, image_path, image, n_orients = 8, ppc = (16,16) , cpb = (4,4) , viz = False, multi_channel = False, normalization = "L2", reduced_color=False, resize = False,
                img_sizeX = 320, img_sizeY = 320):
        # params : n_orients (type = integer)= Number of orientations
        #       : ppc (type = tupel) = Pixels per cell
        #       : cpb (type = tupel) = Cells per block
        #       : viz (type = bool)  = Should visualize hog image
        #       : multi_channel (type = bool) = Is image color or grayscale - gray scale = False
        #       : normalizationn (type = string) = Type of normalization to apply

        hog_image = None

        if image is None:
            if reduced_color:
                image = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_4)

            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        #Convert to BGR if multichannel
        if multi_channel:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if resize:
            image = cv2.resize(image, (img_sizeX, img_sizeY))

        if not viz:
            hog_descriptor = feature.hog(image, orientations = n_orients, pixels_per_cell = ppc,
                     cells_per_block = cpb, visualize = viz, multichannel = multi_channel, block_norm = normalization)

        else :
            hog_descriptor,hog_image = feature.hog(image, orientations=n_orients, pixels_per_cell=ppc,
                                         cells_per_block=cpb, visualize=viz, multichannel=multi_channel,
                                         block_norm=normalization)



        return np.array(hog_descriptor),hog_image

###############################################

    def compute_facenet_embedding_dlib(self, image, upsample = 1, allign = False, draw = False, resize = False, img_sizeX = 320, img_sizeY=320):


        if resize:
            img = cv2.resize(image, (img_sizeX, img_sizeY))

        else:
            img = image




        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self.fd.detect_face(image=img, upsample=upsample)


        # To draw bounding box on the detected face
        if draw:
            color_green = (0, 255, 0)
            line_width = 3

            if self.fd.detector == self.fd.hog_face_detector:
                for det in dets:
                    #if len(dets) > 0:
                    #det = dets[0]
                    cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)

            else:
                for i, d in enumerate(dets):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

                    #cv2.rectangle(img, (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()),
                    #              color_green, line_width)

                    x = d.rect.left()
                    y = d.rect.top()
                    w = d.rect.right() - x
                    h = d.rect.bottom() - y

                    # draw box over face
                    cv2.rectangle(img, (x, y), (x + w, y + h), color_green, line_width)




        print("Number of faces detected: {}".format(len(dets)))

        if len(dets) == 0:
            return None, None


        shapes = self.fd.detect_face_landmarks(dets=dets, image=image, model=self.fd_model)
        descriptors = []


        for shape in shapes:

            # Alligns and crops image 150 x 150 for improved results
            if allign:
                img = dlib.get_face_chip(img, shape)

            d = self.facerec.compute_face_descriptor(img, shape)
            descriptors.append(d)

        return descriptors, img


###############################################


    def compute_facenet_embedding_dlib_batch(self, image_list, batch_size = 128, upsample=1, allign=False, draw=False, resize=False,
                                             img_sizeX=320, img_sizeY=320):

        if resize:
            for i, image in enumerate(image_list):
                image_list[i] = cv2.resize(image, (img_sizeX, img_sizeY))


        dets = self.fd.detect_face_batch(image_list=image_list, batch_size=batch_size, upsample=upsample)



        # To draw bounding box on the detected face
        if draw:
            color_green = (0, 255, 0)
            line_width = 3

            for i, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                    i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
                cv2.rectangle(image_list[i], (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()), color_green, line_width)



        print("Number of faces detected: {}".format(len(dets)))

        if len(dets) == 0:
            return None, None

        for image in image_list:
            shapes = self.fd.detect_face_landmarks(dets=dets, image=image)
            descriptors = []

            for shape in shapes:

                # Alligns and crops image 150 x 150 for improved results
                if allign:
                    img = dlib.get_face_chip(img, shape)

                d = self.facerec.compute_face_descriptor(img, shape)
                descriptors.append(d)

        return descriptors, image_list



###############################################

    #Get the canny edge histogram and the edged image
    def canny_edge_detect_cv2(self, image, lowTh = 90, upTh = 90, sigma = 0.33,  eps=1e-7, auto = False, norm = False, debug = False, blur = False):

        if blur:
            image = cv2.GaussianBlur(image, (5, 5), 0)

        if auto:
            # compute the median of the single channel pixel intensities
            v = np.median(image)

            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            edges = cv2.Canny(image, lower, upper)

        else:
            edges = cv2.Canny(image, lowTh, upTh)

        if debug:
            cv2.imshow("Original", image)
            cv2.imshow("Edged", edges)
            cv2.waitKey(0)

        #Compute the edge histogram
        #hist, bins = np.histogram(edges.ravel(), 256, [0, 256])

        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])

        if norm:
            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

        return hist, edges

###############################################

    def canny_edge_detect_skimage(self, image, plot = False):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(img, (5, 5), 0)
        #edges = feature.canny(imag, sigma=3,low_threshold=0.0375,high_threshold=0.0938)

        edges = feature.canny(img, sigma=1)


        if plot:
            fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                                   figsize=(8, 4))

            ax[0].imshow(image, cmap=plt.cm.gray)
            ax[1].imshow(edges, cmap=plt.cm.gray)

            plt.tight_layout()
            plt.show()


        return edges.ravel()


###############################################

    #Get the SIFT descriptor for the given image
    def compute_sift(self, image, debug = False):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)


        img = cv2.drawKeypoints(gray, kp, image.copy())

        kp, des = sift.compute(gray, kp)

        print(des.shape)

        if debug:
            cv2.imshow("Original", image)
            cv2.imshow("SIFT", img)
            cv2.waitKey(0)

        return  des, img

###############################################


    def get_daisy_feature(self, image, debug = False):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        daisy_desc, daisy_img = feature.daisy(gray, step=100, radius=18, rings=2, histograms=8,
                             orientations=8, visualize=True)

        if debug:
            cv2.imshow("Original", image)
            cv2.imshow("Daisy", daisy_img)
            cv2.waitKey(0)

###############################################



    def compute_hist(image, normalize = True,plot = False, chans = [0], mask = None, binCount = [256], range = [0,256]):

        hist = cv2.calcHist([image], chans, mask, binCount, range)

        if normalize:
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

        if plot:
            plt.plot(hist)
            plt.show()
            plt.xlim([0, 256])


        return  hist


