import cv2 as cv

img=cv.imread('rose.png')
patch=img[250:350,170:270,:]

img=cv.rectangle(img,(170,250),(270,350),(255,0,0),3)
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

cv.imshow('Original',img)
cv.imshow('Resize nearest',patch1)
cv.imshow('Resize bilinear',patch2)
cv.imshow('Resize bicubic',patch3)
def draw(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)
    elif event==cv.EVENT_RBUTTONDOWN:
        cv.rectangle(img,(x,y),(x+200,y+200),(255,0,0),2)

    cv.imshow('Drawing',img)

cv.namedWindow('Drawing')
cv.imshow('Drawing',img)

cv.setMouseCallback('Drawing',draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break

cv.waitKey()
cv.destroyAllWindows()
