from PIL import ImageDraw,Image
import cv2
import numpy as np
import face_recognition
import imutils
import dlib

#select face
image_face = cv2.imread("/Users/hsiehyichin/Desktop/color/people/3.jpg")
image_face = imutils.resize(image_face,width=768,height=1280)            #resize
detector = dlib.get_frontal_face_detector()                              #Dlib偵測臉部
face_rects = detector(image_face,0)                                      #偵測人臉     
for i, d in enumerate(face_rects):                                       #裁切臉
    x1 = d.left()
    y1 = d.top()
    x2 = d.right()
    y2 = d.bottom()
    face = image_face[y1:y2,x1:x2]        
cv2.imwrite("/Users/hsiehyichin/Desktop/color/f/3.bmp",image_face)       #儲存bmp檔
cv2.waitKey(300)
cv2.destroyAllWindows()

#select facial_features
facialfeaturescolor = []
image = face_recognition.load_image_file("/Users/hsiehyichin/Desktop/color/f/3.bmp")
im = Image.open("/Users/hsiehyichin/Desktop/color/f/3.bmp")              #PIL open image
im_rgb = im.convert("RGB")                                               #rgb image
im_gray = im_rgb.convert("L")                                            #gray image
face_landmarks_list = face_recognition.face_landmarks(image)             #point list
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

#facial_features data
#facial_feature -> 五官   face_landmarks[facial_feature] -> 數值

#chin 左右 *掃描不超過
i_chin = len(face_landmarks['chin'])-1
facex = []
facey = []

while i_chin>=0:
    x,y = face_landmarks['chin'][i_chin]
    facex.append(x)
    facey.append(y)
    i_chin = i_chin - 1
    
#lefteyebrow  顏色 反白
i_lefteyebrow = len(face_landmarks['left_eyebrow'])-1
lefteyebrowx = []
lefteyebrowy = []

while i_lefteyebrow>=0:              
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['left_eyebrow'][i_lefteyebrow]
    lefteyebrowx.append(x)
    lefteyebrowy.append(y)
    i_lefteyebrow = i_lefteyebrow-1

#righteyebrow   顏色 反白
i_righteyebrow = len(face_landmarks['right_eyebrow'])-1
righteyebrowx = []
righteyebrowy = []

while i_righteyebrow>=0:
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['right_eyebrow'][i_righteyebrow]
    righteyebrowx.append(x)
    righteyebrowy.append(y)
    i_righteyebrow = i_righteyebrow-1
    
#left_eye       匡出長方形 反白
i_lefteye = len(face_landmarks['left_eye'])-1
lefteyex = []
lefteyey = []

while i_lefteye>=0:
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['left_eye'][i_lefteye]
    lefteyex.append(x)
    lefteyey.append(y)
    i_lefteye = i_lefteye-1
    
#right_eye      匡出長方形 反白
i_righteye = len(face_landmarks['right_eye'])-1
righteyex = []
righteyey = []

while i_righteye>=0:
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['right_eye'][i_righteye]
    righteyex.append(x)
    righteyey.append(y)
    i_righteye = i_righteye-1

#nose
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

#top_lip
i_toplip = len(face_landmarks['top_lip'])-1
toplipx = []
toplipy = []

while i_toplip>=0:
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['top_lip'][i_toplip]
    toplipx.append(x)
    toplipy.append(y)
    i_toplip = i_toplip-1
    
#bottom_lip
i_bottomlip = len(face_landmarks['bottom_lip'])-1
bottomlipx = []
bottomlipy = []

while i_bottomlip>=0:
    color = im_gray.getpixel((x,y))
    facialfeaturescolor.append(color)
    x,y = face_landmarks['bottom_lip'][i_bottomlip]
    bottomlipx.append(x)
    bottomlipy.append(y)
    i_bottomlip = i_bottomlip-1

# mark facial features
image_pil = Image.fromarray(image)
width,height = image_pil.size
d = ImageDraw.Draw(image_pil)     #drawable

d.polygon(face_landmarks['left_eyebrow'], 'white', 'white')
d.polygon(face_landmarks['right_eyebrow'], 'white', 'white')
d.polygon(face_landmarks['left_eye'], 'white', 'white')
d.polygon(face_landmarks['right_eye'], 'white', 'white')

d.polygon((min(nosetipx), max(nosebridgey),
           max(nosetipx), max(nosebridgey),
           max(nosetipx), max(nosetipy),
           min(nosetipx), max(nosetipy)), 'white', 'white')

d.polygon(face_landmarks['top_lip'], 'white', 'white')
# 牙齒
d.polygon(face_landmarks['bottom_lip'], 'white', 'white')
image_pil.save("/Users/hsiehyichin/Desktop/color/adjust/3.bmp")

#adjustment
#automatic
'''scanx = 0
scany = 0
i = 0
while scany < height :
    while scanx > width :
        #這個點的顏色
        color = im_gray.getpixel((scanx,scany))
        #如果和上面的相同會變色
        while i < len(facialfeaturescolor):
            if color == facialfeaturescolor[i]:
                im_rgb.putpixel((scanx,scany),(0,0,0))
            i = i+1
        scanx = scanx + 1
scany = scany + 1'''

#manual
'''def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #點擊左鍵
        pixel = image[y,x]             #取得座標的r,g,b
        print(pixel)
image_adjust = cv2.imread("/Users/hsiehyichin/Desktop/color/adjust/3.bmp")
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", pick_color)
cv2.imshow("Image",image)
cv2.imshow("Adjustment",image_adjust)
cv2.waitKey(3000)
cv2.destroyAllWindows()'''






################################################################################################

#img = cv2.imread('test2.bmp',1)
#height,width,channels = img.shape

#4cmx4cm  ***may cause image damage
#h = int(height/100)
#w = int(width/100)

#cut image
#i = h*w
#x_move = 0
#y_move = 0
'''
while h>0 :
   while w>0 and i>0:
      x_cut = 0+x_move
      y_cut = 0+y_move
      w_cut = 100
      h_cut = 100
      img_cut = img[y_cut:y_cut+h_cut,x_cut:x_cut+w_cut]
      cv2.imwrite('freckle/data'+str(i)+'.bmp',img_cut)
      x_move = x_move+100
      w = w-1
      i = i-1
   w = int(width/25)+1
   y_move = y_move+100
   x_move = 0
   h = h-1
'''
'''
img = cv2.imread('image4.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(3000)
cv2.destroyAllWindows()'''

#read data
'''image_rgb = Image.open("freckle/data4.bmp")    #(r,g,b)
image_gray = image_rgb.convert("L")          #gray
w_data = image_gray.width
h_data = image_gray.height
color_min = 255
x = 0
y = 0
counter = 0

#identify and mark
while h_data>0 :
   
   #find the darkest
   while w_data>0 :
      color = image_gray.getpixel((x,y))
      if color <= color_min:
         color_min = color
      y = y+1
      w_data = w_data-1
      
   # ***mark the darkest in row
   w_data = image_gray.width
   y = 0
   while w_data>0 :
      color = image_gray.getpixel((x,y))
      if color == color_min:
         image_rgb.putpixel((x,y),(255,0,0))
      y = y+1
      w_data = w_data-1
   
   color_min = 255
   w_data = image_gray.width
   h_data = h_data-1
   x = x+1
   y = 0

#save
image_rgb.save('freckle/try.bmp')'''

   

