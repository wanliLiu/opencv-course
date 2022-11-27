import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def root_dir():
    return os.path.abspath(os.path.join(os.path.split(os.path.realpath(__file__))[0], '../'))


def empty(p):
    pass

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",1080,240)
# cv2.createTrackbar("model_selection","TrackBars",100,100,empty)
# cv2.createTrackbar("min_detection_confidence","TrackBars",50,100,empty)


# image = cv2.imread('%s/Resources/Photos/group 2.jpg' % root_dir())
# imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# while True:
#     model_selection = cv2.getTrackbarPos("model_selection", "TrackBars")
#     model_selection = int(model_selection) / 100
#     min_detection_confidence = cv2.getTrackbarPos("min_detection_confidence", "TrackBars")
#     min_detection_confidence = int(min_detection_confidence) / 100
#     face_detection = mp_face_detection.FaceDetection(model_selection=model_selection, min_detection_confidence=min_detection_confidence)
#     # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
#     results = face_detection.process(imageRGB)

#     annotated_image = image.copy()
#     if results.detections:
#         for detection in results.detections:
#             # print('Nose tip:')
#             # print(mp_face_detection.get_key_point(
#             #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
#             mp_drawing.draw_detection(annotated_image, detection)
#         # cv2.imshow('annotated_image', annotated_image)
#     # else:
#         # print("nothing detect")

#     imgStack = stackImages(0.5,([image, imageRGB, annotated_image]))

    
#     cv2.imshow('annotated_image', imgStack)
#     if cv2.waitKey(1) == ord('q'):
#         break

# # For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(1) == ord('q'):
      break
cap.release()