
#main for homo images
import cv2
import numpy as np
import os
i = 0
for filename in os.listdir('/media/chaichana/HDD/WIP/3rd/label/'):
    dst = str(i) + '.jpg'
    path_img = '/media/chaichana/HDD/WIP/3rd/label/' + filename
    #read img opencv
    img = cv2.imread(path_img)
    real = cv2.imread(path_img)
    #convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_range = np.array([140, 9, 131])
    upper_range = np.array([255, 255, 255])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #find piont in squre use Contours
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        area = w * h

        epsilon = 0.08 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if area > 15000:
            cv2.drawContours(img, [approx], -1, (0, 0, 255), 5)
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
            #approx area
            M = approx

            for x in range(0, len(approx)):
                cv2.circle(img, (approx[x][0][0], approx[x][0][1]), 30, (0, 0, 255), -1)

    #protect error when not found area
    try:
        #P is point in squre
        P = [M[0][0][0], M[0][0][1]]
        P1 = [M[1][0][0], M[1][0][1]]
        P2 = [M[2][0][0], M[2][0][1]]
        P3 = [M[3][0][0], M[3][0][1]]
        #Use Homoimg for adjust view img
        pts_src = np.array([P, P1, P2, P3])
        pts_dst = np.array([[0, 0], [0, 400], [600, 400], [600, 0]])
        T, status = cv2.findHomography(pts_src, pts_dst)
        #use real picture for homoimage
        im = cv2.warpPerspective(real, T, (real.shape[1], real.shape[0]))
        Crop_img = im[0:400, 0:600]
        cv2.imwrite('/media/chaichana/HDD/WIP/3rd/Image_Crop/'+dst,Crop_img)
        #improve picture for OCR
        Crop_img = cv2.cvtColor(Crop_img, cv2.COLOR_BGR2GRAY)
        pxmin = np.min(Crop_img)
        pxmax = np.max(Crop_img)
        imgContrast = (Crop_img - pxmin) / (pxmax - pxmin) * 255
        # increase line width
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations=1)
        thresh = cv2.threshold(imgMorph, 150, 255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        result = 255 - opening
        cv2.imwrite('/media/chaichana/HDD/WIP/3rd/Image_OCR/OCR'+dst, result)
        print('Done for '+filename)
        i = i +1
    except:
        print('No found squre in picture')
