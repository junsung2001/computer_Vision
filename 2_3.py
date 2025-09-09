import cv2 as cv
import sys

img=cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #rgb컬러 영상을 변환
gray_small=cv.resize(gray,dsize=(0,0),fx=0.5,fy=0.5) #사이즈 축소

cv.imwrite('soccer_gray.jpg',gray) #파일저장 그레이
cv.imwrite('soccer_gray+small.jpg',gray_small) #파일저장 그레이 스몰

cv.imshow('color image',img)
cv.imshow('gray image',gray)
cv.imshow('gray image small',gray_small)

cv.waitKey()
cv.destroyAllWindows()