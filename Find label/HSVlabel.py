import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
path_img = 'label/Label1111253.jpg'
img = cv2.imread(path_img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([143,0,0])
upper_blue = np.array([255,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
   rect = cv2.boundingRect(c)
   x,y,w,h = rect
   area = w * h

   epsilon = 0.08 * cv2.arcLength(c, True)
   approx = cv2.approxPolyDP(c, epsilon, True)
   if area > 20000:
      cv2.drawContours(img, [approx], -1, (0, 0, 255), 5)
      #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
      print('approx', approx)
      M = approx

      for x in range(0, len(approx)):
         cv2.circle(img, (approx[x][0][0], approx[x][0][1]), 30, (0,0,255), -1)

try:
   P = [M[0][0][0] ,M[0][0][1]]
   P1 = [M[1][0][0] ,M[1][0][1]]
   P2 = [M[2][0][0] ,M[2][0][1]]
   P3 = [M[3][0][0] ,M[3][0][1]]
   pts_src = np.array([P,P1,P2,P3])
   print('src',pts_src)
   pts_dst = np.array([[600, 0],[0, 0],[0,400],[600,400]])
   print('y',pts_dst)
   T, status = cv2.findHomography(pts_src, pts_dst)
   im = cv2.warpPerspective(img, T, (img.shape[1], img.shape[0]))
   Crop_img = im[0:400,0:600]
   Crop_img = cv2.cvtColor(Crop_img, cv2.COLOR_BGR2GRAY)
   pxmin = np.min(Crop_img)
   pxmax = np.max(Crop_img)
   imgContrast = (Crop_img - pxmin) / (pxmax - pxmin) * 255
   # increase line width
   kernel = np.ones((3, 3), np.uint8)
   imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
   cv2.imwrite('out2.png', imgMorph)
   text = pytesseract.image_to_string('out2.png', config='--oem 3 --psm 6')
   print("Detected license plate Number is:",text.replace(" ",""))
   #cv2.drawContours(img, contours, -1, (0,255,0), 5)
   cv2.imshow('im', im)
   cv2.imshow('Crop', Crop_img)
except:
   print('No plate')
img2 = Image.open(path_img)
cv2.imshow('image',img)
cv2.imshow('hsv',mask)
plt.imshow(img2)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
