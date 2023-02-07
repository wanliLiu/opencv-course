import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

img = np.zeros((512,512,3),np.uint8)
#print(img)
#img[:]= 255,0,0



def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)



cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
cv2.circle(img,(400,50),30,(255,255,0),5)
cv2.putText(img," OPENCV 中文 ℃",(0,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),3)
img = cv2AddChineseText(img, "可疑物 32℃", (200, 400))

cv2.imshow("Image",img)

cv2.waitKey(0)