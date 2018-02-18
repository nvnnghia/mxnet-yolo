import os
import cv2
import numpy as np
import sys 
import tools.find_mxnet
import mxnet as mx
import importlib
import KCF
from timeit import default_timer as timer
from detect.detector1 import Detector
import argparse


CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
class Object:
    def __init__(self, xmin1, ymin1, xmax1, ymax1, label1, score):
        self.xmin = xmin1
        self.ymin = ymin1
        self.xmax = xmax1
        self.ymax = ymax1
        self.label = label1
	self.score = score

cap = cv2.VideoCapture('MOT16-01.mp4')
net = None
prefix = '.'
epoch = 0
data_shape = 416
mean_pixels = (123,117,104)
ctx = mx.gpu(0)
batch = 1
objects = []
ret1, frame1 = cap.read()
detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx,batch_size = batch)
def get_batch(imgs):
	    img_len = len(imgs)
	    l = []
	    for i in range(batch):
		if i < img_len:
		    img = np.swapaxes(imgs[i], 0, 2)
		    img = np.swapaxes(img, 1, 2) 
		    img = img[np.newaxis, :] 
		    l.append(img[0])
		else:
		    l.append(np.zeros(shape=(3, data_shape, data_shape)))
	    l = np.array(l)
	    return [mx.nd.array(l)]

a = True
kcf = False
font = cv2.FONT_HERSHEY_SIMPLEX
while(a):
    if(kcf):
	    ret1, framekcf = cap.read()
            
            height,width = framekcf.shape[:2]
            
	    for objecta in (objects):
	        tracker = KCF.kcftracker(False, True, False, False) # hog, fixed_window, multiscale, lab
		tracker.init([objecta.xmin,objecta.ymin,objecta.xmax-objecta.xmin,objecta.ymax-objecta.ymin], framekcf)
		boundingbox = tracker.update(framekcf)
		boundingbox = map(int, boundingbox)
		cv2.rectangle(framekcf,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,0),3)
		cv2.putText(framekcf,objecta.label,(boundingbox[0],boundingbox[1]), font, 1,(0,255,0),2,cv2.LINE_AA)
	    cv2.imshow("img",framekcf)
	    kcf = False
	    objects = []
    else:
	ret, frame = cap.read()
	if (ret == False) :
	    break
	ims = [cv2.resize(frame,(data_shape,data_shape))]

	data  = get_batch(ims)

	start = timer()

	for i in range(1):
	    det_batch = mx.io.DataBatch(data,[])
	    detector.mod.forward(det_batch, is_train=False)
	    detections = detector.mod.get_outputs()[0].asnumpy()
	    result = []
	    
	    for i in range(detections.shape[0]):
		det = detections[i, :, :]
		res = det[np.where(det[:, 0] >= 0)[0]]
		result.append(res)
	    time_elapsed = timer() - start
	    print("Detection time for {} images: {:.4f} sec , fps : {:.4f}".format(1, time_elapsed , (1/time_elapsed)))
	    for k, det in enumerate(result):
		height = frame.shape[0]
		width = frame.shape[1]
		
		for i in range(det.shape[0]):
		    cls_id = int(det[i, 0])
		    if cls_id >= 0:
		        score = det[i, 1]
		        if score > 0.5:
		            xmin = int(det[i, 2] * width)
		            ymin = int(det[i, 3] * height)
		            xmax = int(det[i, 4] * width)
		            ymax = int(det[i, 5] * height)
			    cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),3)
		            class_name = str(cls_id)
		            if CLASSES and len(CLASSES) > cls_id:
		                class_name = CLASSES[cls_id]
			    objecta= Object(xmin, ymin, xmax, ymax, class_name, score)
			    objects.append(objecta)

			    cv2.putText(frame,class_name,(xmin,ymin), font, 1,(0,255,0),2,cv2.LINE_AA)
		cv2.imshow('img',frame)
		kcf = True
    cv2.waitKey(1)
		
cap.release()
cv2.destroyAllWindows()


