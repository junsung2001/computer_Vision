import cv2 as cv

img=cv.imread('mot_color70.jpg') # 영상 읽기
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create() 
kp,des=sift.detectAndCompute(gray,None) #SIFT kp:특징점 의 위치, 크기, 방향 등의 정보 des : 특징점을 설명하는 백터

#특징점 위치와 크기를 원형으로 표시 방향까지 화살표 형태로 시각화
gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift', gray)

k=cv.waitKey()
cv.destroyAllWindows()  