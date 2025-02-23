import cv2
import numpy as np
import os

def root_dir():
    return os.path.split(os.path.realpath(__file__))[0]


def empty(a):
    pass

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

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",1080,240)
cv2.createTrackbar("Hue Min","TrackBars",100,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",100,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",255,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",24,255,empty)
cv2.createTrackbar("Val Max","TrackBars",24,255,empty)

resourceDir = "%s/Resources" % root_dir()
print(f'root-dir = {resourceDir}')

fileList = os.listdir(resourceDir)
fileIndex = 0
# def getContours(img):
#     biggest = np.array([])
#     maxArea = 0
#     contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#        #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
#         peri = cv2.arcLength(cnt,True)
#         approx = cv2.approxPolyDP(cnt,0.02*peri,True)
#         if area > maxArea:
#             maxArea = area
            
#     # cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 40)
#     return biggest


kernel = np.ones((5,5),np.uint8)

# Create the HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    path = os.path.join(resourceDir, fileList[fileIndex])
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    imgBlur = cv2.GaussianBlur(mask,(7,7),0)
    imgCanny = cv2.Canny(imgBlur,150,200)
    imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
    imgEroded = cv2.erode(imgDialation,kernel,iterations=1)

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 10)


    # # Detect the object
    # rects, _ = hog.detectMultiScale(mask, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # # Draw rectangles around the detected objects
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    # cv2.drawContours(imgEroded, contours, -1, (255, 0, 0), 10)
    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)

    imgStack = stackImages(0.3,([img, imgHSV, mask, imgResult],
                                [imgBlur, imgCanny, imgDialation, imgEroded]))
    cv2.imshow("Stacked Images", imgStack)

    key = cv2.waitKey(1)
    if (key == ord('q') ):
        break
    elif (key == ord('n')):
        fileIndex = (fileIndex + 1) % len(fileList)
    elif (key == ord('p')):
        fileIndex -= 1
        if fileIndex < 0:
            fileIndex = len(fileList) - 1