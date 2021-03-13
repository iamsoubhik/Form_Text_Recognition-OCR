import cv2
import pytesseract
import os
import numpy as np

#C:\Program Files\Tesseract-OCR

per = 25
pixelThreshold=500
roi=[[(268, 476), (824, 530), 'text', 'name'],
     [(276, 680), (830, 730), 'text', 'name'],
     [(278, 894), (832, 956), 'text', 'name'],
     [(270, 1124), (828, 1176), 'text', 'name'],
     [(274, 1342), (828, 1392), 'text', 'name']]




pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
imgQ=cv2.imread('Query.png')
h,w,c=imgQ.shape
# imgQ=cv2.resize(imgQ,(w//3,h//3))

orb=cv2.ORB_create(1000) #orb free to use
kp1, des1 = orb.detectAndCompute(imgQ,None) #kp1=keypoints and des1=description
#impKp1=cv2.drawKeypoints(imgQ,kp1,None)

path='UserForms'
myPicList=os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img=cv2.imread(path + "/"+y)
    # img = cv2.resize(img, (w // 3, h // 3))
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) #BRUTFORCE MATCHER
    matches = bf.match(des2,des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch=cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
    # imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0) # FROM DOCUMENTATION MATRIX
    imgScan = cv2.warpPerspective(img, M, (w, h))
    # imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    # cv2.imshow(y, imgScan)
    imgShow=imgScan.copy()
    imgMask=np.zeros_like(imgShow)

    myData=[]

    print(f'***********Extracting data from Form{j}*************')


    for x,r in enumerate(roi):

        cv2.rectangle(imgMask, ((r[0][0]),r[0][1]),((r[1][0]),r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)


    # imgShow = cv2.resize(imgScan, (w // 3, h // 3))


        imgCrop=imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x),imgCrop)

        if r[2] == 'text':
            print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2]=='box':
            imgGray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imageThresh=cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
            totalPixels=cv2.countNonZero(imageThresh)
            if totalPixels>pixelThreshold: totalPixels=1;
            else: totalPixels=0
            print(f'{r[3]}:{totalPixels}')
            myData.append(totalPixels)

        cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

    with open('DataOutput.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')



    imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    print(myData)
    cv2.imshow(y+"2", imgShow)




#cv2.imshow("KeyPointsQuary",impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
