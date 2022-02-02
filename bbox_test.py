import cv2
from cv2 import ROTATE_90_CLOCKWISE

path = './data/codebrim_coco/train/image_0000873.jpg'
bbox = [1997, 520, 1143, 1080]

cv2.namedWindow("bnbox", cv2.WINDOW_NORMAL) 
image = cv2.imread(path)
image = cv2.rotate(image, ROTATE_90_CLOCKWISE)
start_point = (bbox[0], bbox[1])
end_point = (bbox[0]+bbox[2], bbox[1]+bbox[3])
image = cv2.rectangle(image, start_point, end_point, (255,0,0), 5)
cv2.imshow('bnbox',image)
cv2.waitKey()