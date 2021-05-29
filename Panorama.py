import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('pic2.jpeg')
img1 = cv2.resize(img1, (864,1152))
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('pic1.jpeg')
img2 = cv2.resize(img2, (864,1152))
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


sift = cv2.xfeatures2d.SIFT_create()
# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

#show(cv2.drawKeypoints(img1,kp1,None))
#show(cv2.drawKeypoints(img2,kp2,None))
cv2.imshow('right_keypoints',cv2.drawKeypoints(img1,kp1,None))
cv2.imshow('left_keypoints',cv2.drawKeypoints(img2,kp2,None))
cv2.waitKey()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

draw_params = dict(matchColor = (0,255,0), singlePointColor = None, flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

#show(img3)
cv2.imshow("Mathing_lines.jpg", img3)
cv2.waitKey()

MIN_MATCH_COUNT = 15
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)   
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = gray1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)    
    gray2 = cv2.polylines(gray2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    #show(gray2)
    cv2.imshow("image_overlapping.jpg", gray2)
    cv2.waitKey()
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))

dst = cv2.warpPerspective(img1,M,(img2.shape[1] + img1.shape[1], img2.shape[0]))
dst[0:img2.shape[0], 0:img2.shape[1]] = img2
#show(dst)
cv2.imshow("final_before_trimmed.jpg", dst)
cv2.waitKey()

#show(trim(dst))
cv2.imshow("Final.jpg", trim(dst))
cv2.waitKey()
