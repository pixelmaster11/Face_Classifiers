
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
import dlib
from face_detection import FaceDetection
from imutils import paths
import dlib.cuda as cuda
import dlib

'''
This class handles computation of different image and face image encodings
'''



class Face_Encoding:

#########################################################################################################


    def __init__(self, face_detection_model = "HOG", face_landmark_model = "68"):

        cuda.set_device(0)
        dlib.DLIB_USE_CUDA = True
        self.fd_model = face_detection_model
        self.fl_model = face_landmark_model
        self.facerec = dlib.face_recognition_model_v1("Dlib/dlib_face_recognition_resnet_model_v1.dat")
        self.fd = FaceDetection(face_detection_model=face_detection_model , face_landmark_model=face_landmark_model)


#########################################################################################################

    # Compute Local binary pattern of image
    '''
    Params:
        gray - Grayscale image
        numPoints - Number of points used for calculating circular lbp
        radius - Radius of circular lbp computation
        epsilon - For normalization
    
    Returns:
        hist - Computed LBP histogram
        lbp_image - LBP representation of the given image
    '''
    def get_local_binary_pattern(self, gray, numPoints = 24, radius = 8, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns

        lbp = feature.local_binary_pattern(gray, numPoints,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))



        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)


        # Return the histogram of Local Binary Patterns
        return hist, lbp

#########################################################################################################

    # Get color histogram for all color channels
    '''
    Params:
        image - RGB input image
    Returns:
        hist_vector - Computed color (All 3 channels) histograms
        image - Original image
    '''
    def get_color_hist(self, image):
        hist_vector = np.array([])
        bins = 16

        for channel in range(image.shape[2]):
            channel_hist = np.histogram(image[:, :, channel],
                                        bins=bins, range=(0, 255))[0]
            hist_vector = np.hstack((hist_vector, channel_hist))

        #hist = cv2.normalize(hist_vector)
        return hist_vector, image

#########################################################################################################

    #Get spatial histogram of the image
    '''
    Params:
        image - RGB input image
    Returns:
        spatial_vector - Computed spatial image array
        image - Original image
    '''
    def get_spatial_hist(self, image):
        spatial_size = (16, 16)
        spatial_image = cv2.resize(image, spatial_size,
                                   interpolation=cv2.INTER_AREA)
        spatial_vector = spatial_image.ravel()
        return spatial_vector, image

#########################################################################################################

    #Get the hsv color channel histogram
    '''
    Params:
        image - RGB input image
    Returns:
        td_hist - Computed HSV histogram
        image - Original image
    '''
    def get_hsv_hist(self, image):
        bins = 16
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                            [0, 180, 0, 256, 0, 256])

        hist = cv2.normalize(hist)

        # return the flattened histogram as the feature vector
        td_hist = hist.flatten()

        return td_hist, image

#########################################################################################################

    #Get the Histogram of Oriented Gradients for the given  image
    '''
    Params:
        image - RGB image
        image_path - Path from where the image is to be loaded
        n_orients (type = integer)= Number of orientations
        ppc (type = tupel) = Pixels per cell
        cpb (type = tupel) = Cells per block
        multi_channel (type = bool) = Is image color or grayscale - gray scale = False
        normalizationn (type = string) = Type of normalization to apply
        reduced_color - Should the image be of reduced color small size
        resize - Should the image be resized before computing HOG
    
    Returns:
        hog_descriptor - Computed HOG descriptor
        hog_image - HOG representation of the given input image
    '''
    def get_hog(self, image, image_path = None,  n_orients = 8, ppc = (16,16) , cpb = (4,4) , multi_channel = False, normalization = "L2", reduced_color=False, resize = False,
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

        hog_descriptor,hog_image = feature.hog(image, orientations=n_orients, pixels_per_cell=ppc,
                                         cells_per_block=cpb, visualize=True, multichannel=multi_channel,
                                         block_norm=normalization)



        return np.array(hog_descriptor),hog_image

#########################################################################################################

    # Compute the 128-D face embedding for given image
    '''
    Params:
        image - Given RGB/Grayscale image
        dets - Detected face rects
        shapes - Detected face landmarks
        upsample - Whether to upscale the image for better face detection accuracy
        draw - Whether to draw detected face bounding box on the given image
    
    Returns:
        descriptor - A List of descriptors of 128-Dimensions computed from a pretrained (dlib's model) cnn for all detected faces
        image - Original image with or without the face bounding box drawn
    '''
    def compute_facenet_embedding_dlib(self, image, dets = None, shapes = None, upsample = 1, draw = False):


        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.

        if dets is None:
            dets = self.fd.detect_face(image=image, upsample=upsample)


        # To draw bounding box on the detected face
        if draw:
            color_green = (0, 255, 0)
            line_width = 3

            # If HOG based detection used
            if self.fd.detector == self.fd.hog_face_detector:
                for det in dets:
                    cv2.rectangle(image, (det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)

            # Else CNN based detector used
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
                    cv2.rectangle(image, (x, y), (x + w, y + h), color_green, line_width)




        #print("Number of faces detected: {}".format(len(dets)))

        if len(dets) == 0:
            print("No face detected")
            return None, None



        descriptors = []

        # If no landmarks given, detect them
        if shapes is None:
            shapes = self.fd.detect_face_landmarks(dets=dets, image=image)

        # Compute descriptor for all landmarks incase of multiple faces in single image
        #
        for shape in shapes:
            d = self.facerec.compute_face_descriptor(image, shape)
            descriptors.append(d)



        return descriptors, image


#########################################################################################################

    # Compute 128-D face embedding for given batch of image
    '''
    Params:
        image_list - List of images for computing 128-D face embedding
        batch_size - Number of images to be inferrred for batching
        upsample - Number of times to upscale the image for better face detection accuracy
        draw - Whether to draw bounding box or not
    
    Returns:
        descriptor - List of computed 128-D embeddings for given images
        image_list - Given list of images with or without bounding box drawn
    '''
    def compute_facenet_embedding_dlib_batch(self, image_list, batch_size = 128, upsample=1, draw=False):



        dets = self.fd.detect_face_batch(image_list=image_list, batch_size=batch_size, upsample=upsample)



        # To draw bounding box on the detected face
        if draw:
            color_green = (0, 255, 0)
            line_width = 3

            for i, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                    i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
                cv2.rectangle(image_list[i], (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()), color_green, line_width)



        #print("Number of faces detected: {}".format(len(dets)))

        if len(dets) == 0:
            return None, None

        descriptors = []
        for image in image_list:
            shapes = self.fd.detect_face_landmarks(dets=dets, image=image)

            for shape in shapes:
                d = self.facerec.compute_face_descriptor(image, shape)
                descriptors.append(d)

        return descriptors, image_list



#########################################################################################################

    #Get the canny edge histogram and the edged image
    '''
    Params:
        image - Input RGB/Grayscale image
        lowTh - minimum Threshold value
        upThe - maxiimum Threshold value
        sigma - For noise reduction
        aut0 - Whether to automatically calculate max / min threshold values from given sigma value
        Norm - Whether to normalize the histogram
        blur - Whether to apply gaussian blur before computing edges
    
    Returns:
        hist - Edge histogram
        canny - Canny Edge detected image representation
    '''
    def canny_edge_detect_cv2(self, image, lowTh = 30, upTh = 70, sigma = 0.33,  eps=1e-7, auto = False, norm = False, blur = True):

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

        #Compute the edge histogram
        #hist, bins = np.histogram(edges.ravel(), 256, [0, 256])

        #hist = cv2.calcHist([edges],[0], None, [256], [0, 256])
        hist = self.compute_hist(image=edges, chans=[0], mask=None, binCount=[256], range=[0,256], normalize=norm)


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


        return edges
        return edges.ravel(), edges


###############################################

    #Get the SIFT descriptor for the given image
    def compute_sift(self, gray):

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)


        img = cv2.drawKeypoints(gray, kp, image.copy())

        kp, des = sift.compute(gray, kp)

        print(des.shape)

        return  des, img

###############################################


    def get_daisy_feature(self, gray):


        daisy_desc, daisy_img = feature.daisy(gray, step=100, radius=18, rings=2, histograms=8,
                             orientations=8, visualize=True)

        return daisy_desc, daisy_img

###############################################



    def compute_hist(self, image, normalize = True, chans = [0], mask = None, binCount = [256], range = [0,256]):

        hist = cv2.calcHist([image], chans, mask, binCount, range)

        if normalize:
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)

        return  hist


    def plot_hist_color(self, img):


        color = ('b', 'g', 'r')

        for i, col in enumerate(color):

            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])

        plt.show()


    def plot_hist(self, hist):
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()




if __name__ == '__main__':

    fe = Face_Encoding()
    win1 = dlib.image_window()
    win2 = dlib.image_window()
    win3 = dlib.image_window()
    win4 = dlib.image_window()



    for image_path in paths.list_images("D:\Tuts\DataScience\Python\Datasets\FGNET\Age_Test\Old"):
        image = dlib.load_rgb_image(image_path)
        gray = dlib.load_grayscale_image(image_path)

        #image = cv2.resize(image, (300, 300))
        #gray = cv2.resize(gray, (300, 300))

        descriptor, image = fe.compute_facenet_embedding_dlib(image=image, draw=True)

        hist, lbp_image = fe.get_local_binary_pattern(gray=gray)



        hist, hog_image = fe.get_hog(image_path=image_path, image=None, multi_channel=True)

        hist, canny = fe.canny_edge_detect_cv2(image=gray, auto=False, sigma=0.6, lowTh=45, upTh=60)


        contours, heirarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, find the bounding rectangle and draw it
        cv2.drawContours(image, contours, -1, (255, 255, 0), 2, cv2.LINE_AA, maxLevel=1)

        #fe.plot_hist(hist=hist)
        #fe.plot_hist_color(img=image)

        hist, daisy_image = fe.get_daisy_feature(gray=gray)

        win1.set_image(image)
        win2.set_image(hog_image)
        win3.set_image(lbp_image)
        win4.set_image(canny)

        # Use open cv to display daisy images
        #cv2.imshow("Di", daisy_image)
        #cv2.waitKey(0)



        dlib.hit_enter_to_continue()



