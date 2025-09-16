import cv2 as cv
import sys

img=cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다')

cv.imshow('original_RGB',img)
cv.imshow('Upper left half',img[0:img.shape[0]//2,0:img.shape[1]//2,:]) #상단 왼쪽 절반 슬라이싱
cv.imshow('Center half',img[img.shape[0]//4:3*img.shape[0]//4,img.shape[1]//4:3*img.shape[1]//4,:]) #중앙부분 슬라이싱
#img.shape : (높이, 너비, 채널)


cv.imshow('R channel',img[:,:,2])
cv.imshow('G channel',img[:,:,1])
cv.imshow('B channel',img[:,:,0])
#RGB 채널 분리

cv.waitKey()
cv.destroyAllWindows()