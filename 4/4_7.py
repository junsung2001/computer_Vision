import cv2 as cv
import numpy as np

img = cv.imread('soccer.jpg')      
img_show = np.copy(img)             

mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
mask[:] = cv.GC_PR_BGD              # 모든 화소를 배경일 것 같은 값으로 초기화

BrushSiz = 9
LColor, RColor = (255,0,0), (0,0,255)  # 파란색(왼쪽버튼), 빨간색(오른쪽버튼)

def painting(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:   # 왼쪽 버튼 클릭하면 파란색 칠
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)

    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 버튼 클릭하면 빨간색 칠
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)

    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, LColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_FGD, -1)

    elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show, (x,y), BrushSiz, RColor, -1)
        cv.circle(mask, (x,y), BrushSiz, cv.GC_BGD, -1)

    cv.imshow('Painting', img_show)

cv.namedWindow('Painting')
cv.setMouseCallback('Painting', painting)

while(True):
    if cv.waitKey(1) == ord('q'):   # 붓칠을 끝내려면 'q' 키를 누름
        break

background = np.zeros((1,65), np.float64)   # 배경 히스토그램 0으로 초기화
foreground = np.zeros((1,65), np.float64)   # 물체 히스토그램 0으로 초기화

cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK) 
#인수5는 grabcut(영역분할)의 반복횟수로 1로 바꾸면 배경과 물체의 경계가 부자연스럽다

mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
grab = img * mask2[:, :, np.newaxis]
cv.imshow('Grab cut image', grab)

cv.waitKey()
cv.destroyAllWindows()
