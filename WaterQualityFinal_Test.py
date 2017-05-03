#!/usr/bin/python

#1) shrink the image
#-import scipy.misc
#-image = scipy.misc.imresize(image,(400,500))

#2) apply the hough circle transform to mask ev#erything outside of the largest circle (the dish)
#-docs.opencv.org/doc/tutorials/imgproc/imgtrans/hough_circle/hough_circle.html

#3) threshold bacteria by color, using hsv space (he used cymk for some reason)
#-use lemon's hsv_color_picker.py code to find the correct hsv color
#-use lemon's hsv_color_picker.py code to find the correct hsv color
#-color threshold using inrange()

#4) count colonies using a blob detector
#-www.learnopencv.com/blob-detection-using-opencv-python-c/


#########functions#########
import numpy as np
import cv2
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
#import RPi.GPIO as GPIO
#import picamera
import time
import matplotlib.pyplot as plt
import os
#from Tkinter import *


######################## Hough Transform to find the outline of the dish ###########################
#read the image
image = cv2.imread("two_1.jpg")
	   
#print "height,width",image.shape
imageFinale = image.copy()
output = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#shrink the image
[height,width] = gray.shape
oldHeight = height
oldWidth = width
newHeight = int(height/3)
newWidth = int(width/3)
image = scipy.misc.imresize(image,(newHeight,newWidth))
#image1 = scipy.misc.imresize(image,(newHeight,newWidth))
image1 = image.copy()
output = scipy.misc.imresize(output,(newHeight,newWidth))
gray = scipy.misc.imresize(gray,(newHeight,newWidth))

#cv2.imshow("hi",gray)
#cv2.waitKey(0)		
edgeim = cv2.Canny(gray,30,150)
cv2.imshow("output",edgeim)
cv2.waitKey(0)
#plt.show(edgeim)  
#Hough circle transform to label the largest circle in the image (i.e. the plate)
circles = cv2.HoughCircles(edgeim, cv2.HOUGH_GRADIENT, 3.1, 30,minRadius= 10,maxRadius=400)
#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,3.1,10,minRadius=10,maxRadius=50)

#ensure at least some circles were found
if circles is not None:
	print("slide found")

	circles = np.round(circles[0, :]).astype("int")
	x1 = []
	y1 = []
	r1 = []
	for (x, y, r) in circles:
		x1.append(x)
		y1.append(y)
		r1.append(r)	
	x = x1[0]
	y = y1[0]
	r = r1[0]
	
	#if r > 300 or r < 280:
	#	r = 290
	
	#since I know the size of the plates, I can just directly input it once the center has been found.
	
	r = 260
	
	cv2.circle(output, (x, y), r, (0, 255, 0), 4)
	cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	
	#cv2.waitKey(0)
	#print "Outline Found"
	
######################## Hough Transform to find the outline of the dish ################

######################## Using the outline from the Hough Transform to mask everything outside of the dish ################

mask = np.zeros((newHeight,newWidth),dtype="uint8")
white = [255,255,255]
cv2.circle(mask,(x,y),r,white,-1)
masked = cv2.bitwise_and(image, image, mask=mask)


#cv2.imwrite("test1.jpg",masked)
cv2.imshow("output",output)
cv2.waitKey(0)
#print "Dish has been masked"
######################### Using the outline from the Hough Transform to mask everything outside of the dish ##################
loop = 0
while loop<2:
		########## HSV Color detection #############
		#print "HSV Color Detection"
		#image = BGR2HSV(masked)
		image = cv2.cvtColor(masked,cv2.COLOR_BGR2HSV)
		cv2.imshow("image",image)
		cv2.waitKey(0)
		if loop == 0:
				boundaries = [([0,0,50], [40 , 255, 255])]
				#boundaries = [([0,0,50], [18 , 255, 255])]
				#boundaries = [([0,70,40], [18 , 219, 130])]
		else:
				#boundaries = [([20,0,0], [130, 219, 80])]
				#boundaries = [([60,50,20], [100 , 100, 100])]
				#boundaries = [([70,50,50], [75 , 150, 200])]
				boundaries = [([70,50,100], [90 , 150, 130])]
				
		for (lower, upper) in boundaries:
				lower = np.array(lower, dtype = "uint8")
				upper = np.array(upper, dtype = "uint8")
				mask = cv2.inRange(image,lower,upper)
				output = cv2.bitwise_and(image, image, mask = mask)


		print("Color Analysis Completed")
		cv2.imshow("output",output)
		cv2.waitKey(0)
		########## HSV Color detection #############

		########## Convert to the Binary Image ##################
		output = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
		
		
		#group nearby objects to rule out noise later on
		if loop == 1:
			kernel = np.ones((5,5),np.uint8)
			output = cv2.dilate(output,kernel,iterations=4)
			
		if loop == 0:
			kernel = np.ones((3,3),np.uint8)
			output = cv2.dilate(output,kernel,iterations=1)
			
		
		thresh,binaryImage = cv2.threshold(output,0,255,cv2.THRESH_BINARY)
		
		
		cv2.imshow("binaryImage",binaryImage)
		cv2.waitKey(0)
		
		#print "Converted to the Binary Image"
		########## Convert to the Binary Image ##################

		########## Count Using Connected Components ##############
		print("Analyzing...Please Wait")
		dna = binaryImage
		a = np.where(dna > 20)
		b = np.where(dna < 20)
		dna[a] = 255
		dna[b] = 0
		dnaf = dna

		T = 10 
		labeled, nr_objects = ndimage.label(dnaf > T) # `dna[:,:,0]>T` for red-dot case
	   
		#countinue with the algorithm 
		i = 1
		bigger = []
		
		print("step 1")
		while i < nr_objects+1:
				#print "step 1"
				#print i
				TF = np.where(labeled == i)
				nums = dna[TF] 
				#print "nums",numb
				nums = nums.flatten()
				#print "nums.shape",nums.shape
				numslen = len(nums)
				#numslen = nums.shape[0]
				#print "numslen",numslen
				if numslen > 15:
					bigger.append(i)
				i = i+1
		i = 0
		
  
		#while i<len(bigger):
		print("step 2")
		while i<len(bigger):
				#print "step 2"
				num = bigger[i]
				TF = np.where(labeled==num)
				Ymax = np.max(TF[0])
				Ymin = np.min(TF[0])
				Xmax = np.max(TF[1])
				Xmin = np.min(TF[1])		
				dnaBox = dna[Ymin-3:Ymax+3,Xmin-3:Xmax+3]
				kernel = np.ones((3,3),np.uint8)
				#print "loop",loop
				#if loop == 1 and dnaBox is not None:
							#print dnaBox.shape
							#dnaBox = cv2.erode(dnaBox,kernel,iterations = 1)
							#dnaBox = cv2.dilate(dnaBox,kernel,iterations = 3)
				dna[Ymin-3:Ymax+3,Xmin-3:Xmax+3] = dnaBox	
				i = i+1
			
		T > 50
		labeled, nr_objects = ndimage.label(dna > T) 
		i = 1
		bigger = []
		location = []
		print("step 3")
		while i < nr_objects+1:
				#print "step 3"
				TF = np.where(labeled == i)
				nums = dna[TF] 
				nums = nums.flatten()
				#numslen = nums.shape[0]
				numslen = len(nums)
				if loop == 0:
						if numslen > 5 and numslen < 150:
								bigger.append(i)
								Ymax = np.max(TF[0])
								Ymin = np.min(TF[0])
								Xmax = np.max(TF[1])
								Xmin = np.min(TF[1])
								location.append([Ymax-(Ymax-Ymin)/2,Xmax-(Xmax-Xmin)/2])	
								meanBlobX = int(np.mean(TF[0]))	
								meanBlobY = int(np.mean(TF[1]))		
								cv2.circle(image1,(int(meanBlobY/1),int(meanBlobX/1)), 7, (0,0,255),1)
				else:
						if numslen > 5:
								bigger.append(i)
								Ymax = np.max(TF[0])
								Ymin = np.min(TF[0])
								Xmax = np.max(TF[1])
								Xmin = np.min(TF[1])
								location.append([Ymax-(Ymax-Ymin)/2,Xmax-(Xmax-Xmin)/2])
								meanBlobX = int(np.mean(TF[0]))	
								meanBlobY = int(np.mean(TF[1]))		
								cv2.circle(image1,(int(meanBlobY/1),int(meanBlobX/1)), 15, (255,0,0),1)
				i = i+1
		
		if loop == 0:
				print("the number of red colonies is", len(bigger))
				red_number = len(bigger)
			
		else:
				print("the number of blue colonies is", len(bigger))
				blue_number = len(bigger)
	
		######## Count Using Connected Components#########
		
		
		cv2.imwrite("AI_two.jpg",image1)
		
		loop = loop+1
		#print "loop=",loop
			
		
