import cv2
import numpy as np
import cv2.aruco as aruco

def findarucoMarkers(img,markerSize=5,totalMarkers=250,draw=True): #Actually this  function is used to return the BBOXS and ID of respective Aruco Code
    imGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()
    bboxs , ids , rejected= aruco.detectMarkers(imGray,
                                               arucoDict , parameters=arucoParam)

    return [bboxs,ids]



def augmentAruco(bbox , id,approx,img,imgAug , drawid=True):  #just to paste aruco over respective swuare
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    pts1 = np.array([approx[0], approx[1], approx[2], approx[3]])
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))

    pts2=np.array([tl,tr,br,bl])
    matrix, _ = cv2.findHomography(pts2, pts1)

    imgOut=cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    imgOut = img+imgOut
    return imgOut




img=cv2.imread('CVtask.jpg')
img1=cv2.imread('LMAO.jpg')
img2=cv2.imread('XD.jpg')
img3=cv2.imread('Ha.jpg')
img4=cv2.imread('HaHa.jpg')

imGrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_, thrash=cv2.threshold(imGrey,240,255,cv2.THRESH_BINARY)
contours, _= cv2.findContours(thrash, cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx=cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    #cv2.drawContours(img,[approx],0 , (0,0,0),5)
    x=approx.ravel()[0]
    y=approx.ravel()[1]
    if len(approx) ==4:
        x1,y1,w,h=cv2.boundingRect(approx)
        aspectRatio=float(w)/h
        if aspectRatio>=0.95 and aspectRatio<=1.05:

            blue = img[y+30, x+30, 0]
            green = img[y+30, x+30, 1]
            red = img[y+30, x+30, 2]

            if blue>75 and blue <87  and  green>201 and green <211 and  red>145 and red<155:
                arucoFound = findarucoMarkers(img2)
                if len(arucoFound[0]) != 0:
                    for bbox, id in zip(arucoFound[0], arucoFound[1]):
                        img = augmentAruco(bbox, id, approx, img, img1)

            if blue>7 and blue <18 and  green>120 and green <134 and  red>235 and red<247:
                arucoFound = findarucoMarkers(img2)
                if len(arucoFound[0]) != 0:
                    for bbox, id in zip(arucoFound[0], arucoFound[1]):
                        img = augmentAruco(bbox, id, approx, img, img2)

            if blue==0  and  green==0 and  red==0:
                arucoFound = findarucoMarkers(img2)
                if len(arucoFound[0]) != 0:
                    for bbox, id in zip(arucoFound[0], arucoFound[1]):
                        img = augmentAruco(bbox, id, approx, img, img3)
            else:
                arucoFound = findarucoMarkers(img2)
                if len(arucoFound[0]) != 0:
                    for bbox, id in zip(arucoFound[0], arucoFound[1]):
                        img = augmentAruco(bbox, id, approx, img, img4)


#img=cv2.resize(img,(1200,800))  #just for better view of final code as it was getting out of screen
cv2.imwrite('final.jpg',img)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()