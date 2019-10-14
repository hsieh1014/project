from PIL import ImageDraw,Image
from matplotlib import pyplot as plt
import cv2
import numpy as np
import face_recognition
import imutils
import dlib
#cv2 open image
image_jpg = cv2.imread("/Users/hsiehyichin/Desktop/color/people/3.jpg")
#偵測臉部 haarcascades
face = cv2.CascadeClassifier("/Users/hsiehyichin/Desktop/color/xml/haarcascade_frontalface_alt.xml")
#gray image_jpg
gray = cv2.cvtColor(image_jpg, cv2.COLOR_RGB2GRAY)
faces = face.detectMultiScale(gray)
#裁剪臉
for (x, y, w, h) in faces:
    crop_img = image_jpg[y:y+h, x:x+w]
#save face-only image
cv2.imwrite("/Users/hsiehyichin/Desktop/color/img/crop.bmp",crop_img)
#偵測臉部 Dlib
detector = dlib.get_frontal_face_detector()
#偵測人臉
face_rects = detector(image_jpg,0)
#select facial_features
image = face_recognition.load_image_file("/Users/hsiehyichin/Desktop/color/img/crop.bmp")
#PIL open image
im = Image.open("/Users/hsiehyichin/Desktop/color/img/crop.bmp")
#rgb image
image_rgb = im.convert("RGB")
#gray image
image_gray = image_rgb.convert("L")
#drawable image
image_pil = Image.fromarray(image)
width,height = image_pil.size
d = ImageDraw.Draw(image_pil)

#point list
face_landmarks_list = face_recognition.face_landmarks(image)
#facial_features data
#facial_feature -> 五官
#face_landmarks[facial_feature] -> 數值
for face_landmarks in face_landmarks_list:
    facial_features=['chin',
                     'left_eyebrow',
                     'right_eyebrow',
                     'nose_tip',
                     'nose_bridge',
                     'left_eye',
                     'right_eye',
                     'top_lip',
                     'bottom_lip']

#chin匡出臉
i_chin = len(face_landmarks['chin'])-1
facex = []
facey = []
while i_chin>=0:
    x,y = face_landmarks['chin'][i_chin]
    facex.append(x)
    facey.append(y)
    i_chin = i_chin - 1
d.ellipse((min(facex),0,max(facex),max(facey)),outline=(255,255,255), width=0)
image_pil.save("/Users/hsiehyichin/Desktop/color/adjust/3.bmp")

###-------------------------eyebrows-------------------------###
#left
i_lefteyebrow = len(face_landmarks['left_eyebrow'])-1
lefteyebrowx = []
lefteyebrowy = []

while i_lefteyebrow>=0:
    x,y = face_landmarks['left_eyebrow'][i_lefteyebrow]
    lefteyebrowx.append(x)
    lefteyebrowy.append(y)
    i_lefteyebrow = i_lefteyebrow-1

d.polygon(face_landmarks['left_eyebrow'], 'white', 'white')

#right
i_righteyebrow = len(face_landmarks['right_eyebrow'])-1
righteyebrowx = []
righteyebrowy = []

while i_righteyebrow>=0:
    x,y = face_landmarks['right_eyebrow'][i_righteyebrow]
    righteyebrowx.append(x)
    righteyebrowy.append(y)
    i_righteyebrow = i_righteyebrow-1

d.polygon(face_landmarks['right_eyebrow'], 'white', 'white')

###-------------------------eyes-----------------------------###
#left
i_lefteye = len(face_landmarks['left_eye'])-1
lefteyex = []
lefteyey = []

while i_lefteye>=0:
    x,y = face_landmarks['left_eye'][i_lefteye]
    lefteyex.append(x)
    lefteyey.append(y)
    i_lefteye = i_lefteye-1

d.polygon(face_landmarks['left_eye'],'white')

#right
i_righteye = len(face_landmarks['right_eye'])-1
righteyex = []
righteyey = []

while i_righteye>=0:
    x,y = face_landmarks['right_eye'][i_righteye]
    righteyex.append(x)
    righteyey.append(y)
    i_righteye = i_righteye-1

d.polygon(face_landmarks['right_eye'],'white')

###-------------------------nose-----------------------------###
i_nosetip = len(face_landmarks['nose_tip'])-1
i_nosebridge = len(face_landmarks['nose_bridge'])-1
nosetipx = []
nosetipy = []
nosebridgex = []
nosebridgey = []

while i_nosetip>=0:
    x,y = face_landmarks['nose_tip'][i_nosetip]
    nosetipx.append(x)
    nosetipy.append(y)
    i_nosetip = i_nosetip-1

while i_nosebridge>=0:
    x,y = face_landmarks['nose_bridge'][i_nosebridge]
    nosebridgex.append(x)
    nosebridgey.append(y)
    i_nosebridge = i_nosebridge-1

d.polygon((min(nosetipx), max(nosebridgey),
           max(nosetipx), max(nosebridgey),
           max(nosetipx), max(nosetipy),
           min(nosetipx), max(nosetipy)),'white')

###-------------------------mouth----------------------------###
#tensorflow
#top_lip
i_toplip = len(face_landmarks['top_lip'])-1
toplipx = []
toplipy = []

while i_toplip>=0:
    x,y = face_landmarks['top_lip'][i_toplip]
    toplipx.append(x)
    toplipy.append(y)
    i_toplip = i_toplip-1

#bottom_lip
i_bottomlip = len(face_landmarks['bottom_lip'])-1
bottomlipx = []
bottomlipy = []

while i_bottomlip>=0:
    x,y = face_landmarks['bottom_lip'][i_bottomlip]
    bottomlipx.append(x)
    bottomlipy.append(y)
    i_bottomlip = i_bottomlip-1

d.polygon((min(toplipx),min(toplipy),
           max(toplipx),min(toplipy),
           max(bottomlipx),max(bottomlipy),
           min(bottomlipx),max(bottomlipy)),'white')
image_pil.save("/Users/hsiehyichin/Desktop/color/adjust/3.bmp")

#cut
######如果不是倍數切割會出現不能打開的檔案
img = cv2.imread("/Users/hsiehyichin/Desktop/color/adjust/3.bmp",1)

h = int(height/100)
w = int(width/100)
w_cut = 100
h_cut = 100
i = h*w
x_move = 0
y_move = 0
while h > 0 :
   while w > 0 :
       x_cut = 0+x_move
       y_cut = 0+y_move
       img_cut = img[y_cut:y_cut+h_cut,x_cut:x_cut+w_cut]
       cv2.imwrite('/Users/hsiehyichin/Desktop/color/freckle/data'+str(i)+'.bmp',img_cut)
       x_move = x_move+100
       w = w-1
       i = i-1
   w = int(width/25)+1
   y_move = y_move+100
   x_move = 0
   h = h-1
