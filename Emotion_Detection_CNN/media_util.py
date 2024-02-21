import cv2
img = cv2.imread('C:/Users/Admin/Documents/myProject/Emotion_Detection_CNN/emojis/angry.png',-1)
print(img.shape)
cv2.imshow('Display Image', img)
cv2.waitKey(0)
#cv2.imshow('image', img)