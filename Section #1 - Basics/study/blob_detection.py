import cv2 as cv
import numpy as np
import os

def root_dir():
    return os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..'))


img = cv.imread('%s/Resources/Photos/lady.jpg' % root_dir())
# cv.imshow("gray scale image", img)


# Setup SimpleBlobDetector parameters.
params = cv.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = False
params.minArea = 1000
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
 
# Create a detector with the parameters
ver = (cv.__version__).split('.')
if int(ver[0]) < 3 :
  detector = cv.SimpleBlobDetector(params)
else : 
  detector = cv.SimpleBlobDetector_create(params)

keypoints = detector.detect(img)


im_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("Keypoints", im_with_keypoints)


cv.waitKey(0)