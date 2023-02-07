import cv2
import numpy as np
import os


def root_dir():
    return os.path.split(os.path.realpath(__file__))[0]

resourceDir = "%s/Resources" % root_dir()
print(f'root-dir = {resourceDir}')

# 读取图像
img_input = cv2.imread(os.path.join(resourceDir, "fir_2022_11_28_10_07_01_255.png"))

img = cv2.pyrDown(img_input)
# 创建HOG检测器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 执行HOG检测
boxes, weights = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)

# 在图像中标记识别到的物体
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

# 显示结果
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
