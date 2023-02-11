# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:35:06 2020

@author: zqq
"""

import numpy as np

boxes=np.array([
        [100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,190,200,0.71],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]]) 
 
 
def py_cpu_nms(dets, thresh):
    "Pure Python NMS baseline"
    # x1、y1、x2、y2以及score赋值
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:, 4]

    # 每一个检测框的面积
    areas = (y2-y1+1) * (x2-x1+1)
    print(areas)
    # 按照score置信度降序排序
    order = scores.argsort()[::-1]

    keep = [] # 保留的结果框集合
    while order.size >0:
        i = order[0]       # every time the first is the biggst, and add it directly
        keep.append(i) # 保留该类剩余box中得分最高的一个
        # 得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算相交的面积,不重叠时面积为0
        w = np.maximum(0, xx2-xx1+1)    # the weights of overlap
        h = np.maximum(0, yy2-yy1+1)    # the height of overlap
        inter = w*h
        # 计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i]+areas[order[1:]] - inter)
        # 保留IoU小于阈值的box
        indx = np.where(ovr<=thresh)[0]
        order = order[indx+1]   # 因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位

    return keep
        
 
import matplotlib.pyplot as plt
def plot_bbox(dets, c='k'):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    
    plt.plot([x1,x2], [y1,y1], c)
    plt.plot([x1,x1], [y1,y2], c)
    plt.plot([x1,x2], [y2,y2], c)
    plt.plot([x2,x2], [y1,y2], c)
    #plt.title(" nms")
    #plt.show()

plt.figure(1)
ax1 = plt.subplot(1,2,1)
ax1.set_title('before nms')
ax2 = plt.subplot(1,2,2)
ax2.set_title('after nms')
 
plt.sca(ax1)
plot_bbox(boxes,'k')   # before nms

keep = py_cpu_nms(boxes, thresh=0.2)
plt.sca(ax2)
plot_bbox(boxes[keep], 'b')# after nms
plt.show()
