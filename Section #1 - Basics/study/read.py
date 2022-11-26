import cv2 as cv

import os

def root_dir():
    return os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../..'))

def rescaleFrame(frame, scale=.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions,interpolation=cv.INTER_AREA)


img = cv.imread('%s/Resources/Photos/cat_large.jpg' % root_dir())

gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
cv.imshow('gray_img', gray_img)

img_rescale = rescaleFrame(img, scale=.2)
cv.imshow('Cat', img)
cv.imshow('cat resig', img_rescale)
# cv.waitKey(0)



capture = cv.VideoCapture("%s/Resources/Videos/dog.mp4" % root_dir())
# capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if isTrue:
        frame_resize = rescaleFrame(frame, scale=.2)
        cv.imshow('video', frame)
        cv.imshow('video resize', frame_resize)
        if cv.waitKey(20) & 0xff == ord('d'):
            break;
    else:
        break


capture.release()
cv.destroyAllWindows()