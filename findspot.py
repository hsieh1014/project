#coding=utf-8
from PIL import ImageDraw,Image,ImageTk,ImageGrab
import imutils
import dlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
import tensorflow as tf
import sys
#This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
import label_map_util
import visualization_utils as vis_util
import tkinter as tk
from tkinter import filedialog
import shutil
import setup
import face_recognition
#initialize
shutil.rmtree("/Users/hsiehyichin/Desktop/project/result")
shutil.rmtree("/Users/hsiehyichin/Desktop/project/color")
os.mkdir("/Users/hsiehyichin/Desktop/project/result")
os.mkdir("/Users/hsiehyichin/Desktop/project/color")
#input image tk window
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
#cv2 open image
image_jpg = cv2.imread(file_path)
#detect facehaarcascades
face = cv2.CascadeClassifier("/Users/hsiehyichin/Desktop/project/xml/haarcascade_frontalface_alt.xml")
#gray image_jpg
gray = cv2.cvtColor(image_jpg,cv2.COLOR_RGB2GRAY)
faces = face.detectMultiScale(gray)
#cutting face
for (x, y, w, h) in faces:
    image_cut = image_jpg[y:y+h, x:x+w]
    #resize
    image_cut = cv2.resize(image_cut,(1000,1000),interpolation=cv2.INTER_CUBIC)
#save face-only image
cv2.imwrite("/Users/hsiehyichin/Desktop/project/cut/cut.bmp",image_cut)
height,width,channels = image_cut.shape
#detect face Dlib
detector = dlib.get_frontal_face_detector()
face_rects = detector(image_cut,0)
#select facial_features
image_face = face_recognition.load_image_file("/Users/hsiehyichin/Desktop/project/cut/cut.bmp")
#point list
face_landmarks_list = face_recognition.face_landmarks(image_face)
#facial_features data
#facial_feature -> 五官
#face_landmarks[facial_feature] -> 數值
for face_landmarks in face_landmarks_list:
    facial_features=['chin',
                     'left_eyebrow','right_eyebrow',
                     'nose_tip','nose_bridge',
                     'left_eye','right_eye',
                     'top_lip','bottom_lip']

#chin -> Cutting face
i = len(face_landmarks['chin'])
count = len(face_landmarks['chin'])-1
facex = np.zeros(i,dtype=np.int)
facey = np.zeros(i,dtype=np.int)
h = int(height/2)
while count>=0:
    x,y = face_landmarks['chin'][count]
    facex[count] = x
    facey[count] = y
    count = count - 1
w_max = max(facex)
w_min = min(facex)
center_x  = int(((w_max+w_min)/2))
#circle the face field
blackmask = np.zeros(image_cut.shape[:2], dtype=np.uint8)
blackmask = cv2.ellipse(blackmask,(center_x,h),((center_x-w_min),h),0,0,360,(255,255,255),-1)

#show -> mark
image_show = image_cut.copy()
#cut  -> save & analyze
image_cut = cv2.add(image_cut,np.zeros(np.shape(image_cut),dtype=np.uint8),mask=blackmask)
cv2.imwrite("/Users/hsiehyichin/Desktop/project/cut/cut.bmp",image_cut)
#data -> adjust
image_data = Image.open("/Users/hsiehyichin/Desktop/project/cut/cut.bmp").convert('RGBA')
base = Image.new('RGBA',(width,height),color = (255,255,255,0))
d = ImageDraw.Draw(base)

#eyebrows + eyes
eye_part = ['left','right']
for eye in eye_part:
    count= len(face_landmarks[eye+'_eyebrow'])-1
    eyebrow_x_min = width
    eyebrow_y_min = height
    while count>= 0:
        x,y = face_landmarks[eye+'_eyebrow'][count]
        if x < eyebrow_x_min:
            eyebrow_x_min = x
        if y < eyebrow_y_min:
            eyebrow_y_min = y
        count= count-1
    count = len(face_landmarks[eye+'_eye'])-1
    eye_x_max = 0
    eye_x_min = width
    eye_y_max = 0
    while count >= 0:
        x,y = face_landmarks[eye+'_eye'][count]
        if x > eye_x_max:
            eye_x_max = x
        elif x < eye_x_min:
            eye_x_min = x
        if y > eye_y_max:
            eye_y_max = y
        count = count-1
    center_x = int((eye_x_max+eye_x_min)/2)
    center_y = int((eye_y_max+eyebrow_y_min)/2)+5
    ax = center_x-eyebrow_x_min
    ay = center_y-eyebrow_y_min
    cv2.ellipse(image_cut,(center_x,center_y),(ax+10,ay+10),0,0,360,(0,0,0),-1)
    d.ellipse([(center_x-ax-10,center_y-ay-10),(center_x+ax+10,center_y+ay+10)],fill=(255,0,0,100),outline=None)

#nose
count = len(face_landmarks['nose_tip'])-1
nosetip_y_max = 0
nosetip_x_min = width
nosetip_x_max = 0
while count>=0:
    x,y = face_landmarks['nose_tip'][count]
    if y > nosetip_y_max:
        nosetip_y_max = y
    if x > nosetip_x_max:
        nosetip_x_max = x
    elif x < nosetip_x_min:
        nosetip_x_min = x
    count = count-1
count = len(face_landmarks['nose_bridge'])-1
nosebridge_y_max = 0
while count>=0:
    x,y = face_landmarks['nose_bridge'][count]
    if y > nosebridge_y_max:
        nosebridge_y_max = y
    count = count-1
cv2.rectangle(image_cut,(nosetip_x_min-2,nosebridge_y_max),(nosetip_x_max,nosetip_y_max),(0,0,0),-1)
d.rectangle([(nosetip_x_min-2,nosebridge_y_max),(nosetip_x_max,nosetip_y_max)],fill=(255,0,0,100),outline=None)

#cut into two part
h = nosebridge_y_max
for i in range(2,h):
       if (h % i) == 0:
           break
       else:
           h = h-1
           break
w = int((w_max-w_min)/2)
for i in range(2,w):
       if (w % i) == 0:
           break
       else:
           w_max = w_max-1
           break

#調整用
image_data = image_data.crop((w_min,0,w_max,height))
base = base.crop((w_min,0,w_max,height))
out = Image.alpha_composite(image_data,base)
out.save("/Users/hsiehyichin/Desktop/project/image/adjust.bmp")
#分析用
image_up = image_cut[0:h,w_min:w_max]
cv2.imwrite("/Users/hsiehyichin/Desktop/project/image/data.bmp",image_up)
image_down = image_cut[h:1000,w_min:w_max]
cv2.imwrite("/Users/hsiehyichin/Desktop/project/image/datadown.bmp",image_down)

#標示用
image_show = image_show[0:1000,w_min:w_max]
w = int((w_max-w_min)/2)
image_show = cv2.resize(image_show,(w,500))
cv2.imwrite("/Users/hsiehyichin/Desktop/project/show/show.bmp",image_show)

#原圖
image_input = image_show.copy()
cv2.imwrite("/Users/hsiehyichin/Desktop/project/show/input.bmp",image_input)


#mouth -> tensorflow
MODEL_NAME = "/Users/hsiehyichin/Desktop/project/venv/tensorflow1/models/research/object_detection/inference_graph"
IMAGE_NAME = "/Users/hsiehyichin/Desktop/project/image/datadown.bmp"
#Grab path to current working directory
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
#Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
NUM_CLASSES = 1
#Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with  tf.io.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)
#Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#Output tensors are the detection boxes, scores, and classes
#Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#Each score represents level of confidence for each of the objects.
#The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#Load image using OpenCV and
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)
#Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})
#Draw the results of the detection
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=1,
    min_score_thresh=0.60)
cv2.imwrite("/Users/hsiehyichin/Desktop/project/image/datadown.bmp",image)

#merge
image1 = Image.open("/Users/hsiehyichin/Desktop/project/image/data.bmp")
image2 = Image.open("/Users/hsiehyichin/Desktop/project/image/datadown.bmp")
image_new = Image.new('RGB',(image1.width,image1.height+image2.height))
image_new.paste(image1,(0,0))
image_new.paste(image2,(0,image1.height))
############################
image_new = image_new.resize((w,500))
image_new.save("/Users/hsiehyichin/Desktop/project/image/data.bmp")

'''#adjust
#click -> True
drawing = False
#if true then draw line
mode = True
ix, iy = -1, -1
def draw_line(event,x,y,flags,param):
    global ix,iy, drawing, mode
    #click
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    #move
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(image_adjust,(ix,iy),(x,y),(153,153,255),2)
                cv2.line(image,(ix,iy),(x,y),(0,0,0),2)
    #unclick
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

image = cv2.imread("/Users/hsiehyichin/Desktop/project/image/data.bmp")
image_adjust = cv2.imread("/Users/hsiehyichin/Desktop/project/image/adjust.bmp")
image = cv2.resize(image,(w,500),interpolation=cv2.INTER_CUBIC)
image_adjust = cv2.resize(image_adjust,(w,500),interpolation=cv2.INTER_CUBIC)
cv2.namedWindow("face")
cv2.moveWindow("face",0,0)
#cv2.imshow("face",image)
cv2.setMouseCallback("face",draw_line)
while(1):
    cv2.imshow("face",image_adjust)
    k = cv2.waitKey(1) & 0xFF
    #click m -> can't draw
    if k == ord("m"):
        mode = not mode
    #Esc -> escape
    elif k == 27:
        cv2.imwrite("/Users/hsiehyichin/Desktop/project/image/data.bmp",image)
        break
cv2.destroyAllWindows()'''

#analyze
def showMaxFactor(num):
    a = []
    count = num
    while count > 1:
        if num % count == 0:
            a.append(count)
        count = count-1
    return a

#mark
image_show = Image.open("/Users/hsiehyichin/Desktop/project/show/show.bmp")
#cv2 open image
image = Image.open("/Users/hsiehyichin/Desktop/project/image/data.bmp")
width,height = image.size
factor_y = showMaxFactor(height)
factory = factor_y[3] #100
factor_x = showMaxFactor(width)
factorx  = factor_x[1] #133
i_y = int(height/factory) #5
i_x = int(width/factorx) #3
level1 = []
level2 = []
level3 = []
x = 0
y = 0
while i_y>0:
    while i_x>0:
        #mark
        img_cut_2 = image_show.crop((x,y,x+factorx,y+factory))
        img_cut_2.save("/Users/hsiehyichin/Desktop/project/result/data"+str(int(i_y))+str(i_x)+".bmp")
        #color
        img_cut_1 = image.crop((x,y,x+factorx,y+factory))
        img_cut_1.save("/Users/hsiehyichin/Desktop/project/color/data"+str(int(i_y))+str(i_x)+".bmp")
        #gray
        image_gray = img_cut_1.convert("L")
        #drawable
        draw = ImageDraw.Draw(img_cut_2)
        width_g = image_gray.width
        height_g = image_gray.height
        allcolor = []
        mode = []
        y_r = 0
        x_r = 0
        #make all color list
        while y_r < height_g:
            while x_r < width_g:
                color = image_gray.getpixel((x_r,y_r))
                if color != 255 and color != 0:
                    allcolor.append(color)
                x_r = x_r+5
            y_r = y_r+5
            x_r = 0 #initialize
        #find mode
        color_appear = dict((a,allcolor.count(a)) for a in allcolor)
        if max(color_appear.values()) == 1:
            for k,v in color_appear.items():
                modei = int(max(color_appear.keys())+min(color_appear.keys()))
        else:
            #k is key,and v is value
            for k,v in color_appear.items():
                if v == max(color_appear.values()):
                    #collect mode
                    mode.append(k)
            modei = min(mode)
        #darker than average color list
        darkercolor = []
        for a in color_appear.keys():
            if a < modei:
                darkercolor.append(a)
        darkercolor = list(set(darkercolor))
        d = int(len(darkercolor)/3)
        dark = min(darkercolor)
        y_r = 5 #initialize
        i_r = 1
        while y_r < height_g:
            #initialize
            if i_r%2 == 0:
                x_r = 0
            else:
                x_r = 5
            while x_r < width_g:
                color = image_gray.getpixel((x_r,y_r))
                if color>=dark and color<(dark+d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='red')
                    level1.append((x_r,y_r))
                elif color>=(dark+d) and color<(dark+2*d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='yellow')
                    level2.append((x_r,y_r))
                elif color>=(dark+2*d) and color<(dark+3*d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='blue')
                    level3.append((x_r,y_r))
                i_r = i_r+1
                x_r = x_r+10
            y_r = y_r+10
        img_cut_2.save("/Users/hsiehyichin/Desktop/project/result/data"+str(int(i_y))+str(i_x)+".bmp")
        x = x+factorx
        i_x = i_x-1
    y = y+factory
    #initialize
    x = 0
    i_x = int(width/factorx)
    i_y = i_y-1

computercount = len(level1)+len(level2)+len(level3)

#merge multiple images
#initialize
i_y = int(height/factory)
i_x = int(width/factorx)
i = 0
while i_y > 0:
    while i_x > 1:
        image1 = Image.open("/Users/hsiehyichin/Desktop/project/result/data"+str(int(i_y))+str(i_x)+".bmp")
        i_x = i_x-1
        image2 = Image.open("/Users/hsiehyichin/Desktop/project/result/data"+str(int(i_y))+str(i_x)+".bmp")
        image_new = Image.new("RGB",(image1.width+image2.width,image1.height))
        image_new.paste(image1,(0,0))
        image_new.paste(image2,(image1.width,0))
        image_new.save("/Users/hsiehyichin/Desktop/project/result/data"+str(int(i_y))+str(i_x)+".bmp")
    i = i+1
    i_x = int(width/factorx)
    i_y = i_y-1
while i > 1:
    image1 = Image.open("/Users/hsiehyichin/Desktop/project/result/data"+str(i)+"1.bmp")
    i = i-1
    image2 = Image.open("/Users/hsiehyichin/Desktop/project/result/data"+str(i)+"1.bmp")
    image_new = Image.new('RGB',(image1.width,image1.height+image2.height))
    image_new.paste(image1,(0,0))
    image_new.paste(image2,(0,image1.height))
    image_new.save("/Users/hsiehyichin/Desktop/project/result/data"+str(i)+"1.bmp")

root.destroy()

#frame
root1 = tk.Tk()
root1.title("result")
root1.geometry("1280x800")
canvas_back = tk.Canvas(root1,width=1280,height=800,bd=-2)
canvas_back.pack()
canvas_back.background = ImageTk.PhotoImage(Image.open("/Users/hsiehyichin/Desktop/project/back.jpg"))
canvas_back.create_image(0,0,anchor='nw',image=canvas_back.background)
level1_g = []
level2_g = []
level3_g = []

#left -> grab image to analyze
#label
var1 = tk.StringVar()
var1.set("原本的輸入")
l = tk.Label(root1,textvariable=var1,background="#F28482",foreground="#F7EDE2",font=("Helvetica",20)).place(x=275,y=670,anchor='nw')
spot = tk.Label(root1,text="斑點數目",background="#427AA1",foreground="#EBF2FA",font=("Helvetica",20)).place(x=10,y=450,anchor='nw')
var4 = tk.StringVar()
var4.set("0")
spotnum = tk.Label(root1,textvariable=var4,background="#427AA1",foreground="#EBF2FA",font=("Helvetica",20)).place(x=40,y=490,anchor='nw')
class App(tk.Frame):
    def __init__(self,parent):
        tk.Frame.__init__(self,parent)
        self._createVariables(parent)
        self._createCanvas()
        self._createCanvasBinding()

    def _createVariables(self, parent):
        self.parent = parent
        self.rectx0 = 0
        self.recty0 = 0
        self.rectx1 = 0
        self.recty1 = 0
        self.rectid = None

    def _createCanvas(self):
        self.img = Image.open("/Users/hsiehyichin/Desktop/project/show/show.bmp")
        width,height = self.img.size
        self.canvas = tk.Canvas(root1,bg="white",height=height,width=width,cursor="plus")
        w = int((640-width)/2)
        self.canvas.place(x=w+20,y=120,anchor='nw')
        self.canvas.image = ImageTk.PhotoImage(self.img)
        self.canvas.create_image(0,0,image=self.canvas.image,anchor='nw')

    def _createCanvasBinding(self):
        self.canvas.bind("<Button-1>",self.startRect)
        self.canvas.bind("<ButtonRelease-1>",self.stopRect)
        self.canvas.bind("<B1-Motion>",self.movingRect)

    def startRect(self, event):
        self.rectx0 = self.canvas.canvasx(event.x)
        self.recty0 = self.canvas.canvasy(event.y)
        #Create rectangle
        self.rectid = self.canvas.create_rectangle(self.rectx0, self.recty0, self.rectx0, self.recty0,outline="#000000")

    def movingRect(self, event):
        self.rectx1 = self.canvas.canvasx(event.x)
        self.recty1 = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rectid, self.rectx0, self.recty0,self.rectx1, self.recty1)


    def stopRect(self, event):
        self.rectx1 = self.canvas.canvasx(event.x)
        self.recty1 = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rectid, self.rectx0, self.recty0,self.rectx1, self.recty1)
        #analyze
        self.img2 = Image.open("/Users/hsiehyichin/Desktop/project/image/data.bmp")
        self.img2 = self.img2.crop((self.rectx0,self.recty0,self.rectx1,self.recty1))
        image_gray = self.img2.convert("L")
        width_g = image_gray.width
        height_g = image_gray.height
        #mark and show
        self.img3 = self.img.copy()
        self.img3 = self.img3.crop((self.rectx0,self.recty0,self.rectx1,self.recty1))
        draw = ImageDraw.Draw(self.img3)
        allcolor = []
        mode = []
        y_r = 0
        x_r = 0
        #make all color list
        while y_r < height_g:
            while x_r < width_g:
                color = image_gray.getpixel((x_r,y_r))
                if color != 255 and color != 0:
                    allcolor.append(color)
                x_r = x_r+2
            y_r = y_r+2
            x_r = 0 #initialize
        #find mode
        color_appear = dict((a,allcolor.count(a)) for a in allcolor)
        if max(color_appear.values()) == 1:
            for k,v in color_appear.items():
                modei = int(max(color_appear.keys())+min(color_appear.keys()))
        else:
            #k is key,and v is value
            for k,v in color_appear.items():
                if v == max(color_appear.values()):
                    #collect mode
                    mode.append(k)
            modei = min(mode)
        #darker than average color list
        darkercolor = []
        for a in color_appear.keys():
            if a < modei:
                darkercolor.append(a)
        darkercolor = list(set(darkercolor))
        d = int(len(darkercolor)/3)
        dark = min(darkercolor)
        y_r = 5 #initialize
        i_r = 1
        while y_r < height_g:
            #initialize
            if i_r%2 == 0:
                x_r = 0
            else:
                x_r = 5
            while x_r < width_g:
                color = image_gray.getpixel((x_r,y_r))
                if color>=dark and color<(dark+d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='red')
                    level1_g.append((int(x_r+self.rectx0),int(y_r+self.recty0)))
                elif color>=(dark+d) and color<(dark+2*d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='yellow')
                    level2_g.append((int(x_r+self.rectx0),int(y_r+self.recty0)))
                elif color>=(dark+2*d) and color<(dark+3*d):
                    draw.ellipse((x_r-5,y_r-5,x_r+5,y_r+5),fill='white',outline='blue')
                    level3_g.append((int(x_r+self.rectx0),int(y_r+self.recty0)))
                i_r = i_r+1
                x_r = x_r+10
            y_r = y_r+10

        #merge with canvas
        image1 = Image.open("/Users/hsiehyichin/Desktop/project/show/show.bmp")
        image_new = Image.new('RGB',(image1.width,image1.height))
        image_new.paste(image1,(0,0))
        image_new.paste(self.img3,(int(self.rectx0),int(self.recty0)))
        image_new.save("/Users/hsiehyichin/Desktop/project/show/show.bmp")
        self.canvas.image = ImageTk.PhotoImage(image_new)
        self.canvas.create_image(0,0,image=self.canvas.image,anchor='nw')

#button
on_hit = False
def makecanvas():
    global on_hit
    #hit
    if on_hit == False:
        on_hit = True
        app = App(root1)
        var1.set("擷取的部分")
    #not hit
    else:
        on_hit = False

def showresult_o():
    global on_hit
    #hit
    if on_hit == False:
        on_hit = True
        canvas_ui = tk.Canvas(root1,bg="white",height=height,width=width)
        canvas_ui.place(x=w+20,y=120,anchor='nw')
        canvas_ui.image = ImageTk.PhotoImage(img_input)
        canvas_ui.create_image(0,0,image=canvas_ui.image,anchor='nw')
        var1.set("原本的輸入")
    #not hit
    else:
        on_hit = False

#pil load image
img_input = Image.open("/Users/hsiehyichin/Desktop/project/show/input.bmp")
width,height = img_input.size
canvas_ui = tk.Canvas(root1,bg="white",height=height,width=width)
w = int((640-width)/2)
canvas_ui.place(x=w+20,y=120,anchor='nw')
canvas_ui.image = ImageTk.PhotoImage(img_input)
canvas_ui.create_image(0,0,image=canvas_ui.image,anchor='nw')
#grab partial image Button
imgBtn = tk.PhotoImage(file="/Users/hsiehyichin/Desktop/project/icons8-screenshot-50.png")
b = tk.Button(root1,image=imgBtn,command=makecanvas,font="helvetica")
b.place(x=w+width+82,y=200,anchor='nw')
b3 = tk.Button(root1,fg="#847577",text="原圖",command=showresult_o,font="helvetica",width=5,height=2)
b3.place(x=w+width+90,y=270,anchor='nw')
drawpoint = []

#right -> output cumputer analyze and grab result
#button -> computer
on_hit = False
def showresult_c():
    global on_hit
    #hit
    if on_hit == False:
        on_hit = True
        vlabel.configure(image=imtk)
        vlabel.photo = imtk
        var2.set("電腦的模擬")
        var4.set(computercount)
    #not hit
    else:
        on_hit = False

#button -> grab
def showresult_g():
    global on_hit
    #hit
    if on_hit == False:
        on_hit = True
        img = Image.open("/Users/hsiehyichin/Desktop/project/show/show.bmp")
        imtk_g = ImageTk.PhotoImage(img)
        vlabel.configure(image=imtk_g)
        vlabel.photo = imtk_g
        var2.set("擷取的分析")
        partiallycount = len(level1_g)+len(level2_g)+len(level3_g)
        var4.set(partiallycount)
    #not hit
    else:
        on_hit = False

def l1():
    global on_hit
    global drawpoint
    #hit
    if on_hit == False:
        on_hit = True
        drawpoint = level1_g
        img_select1 = img_input.copy()
        imtk_i = ImageTk.PhotoImage(img_select1)
        draw_i= ImageDraw.Draw(img_select1)
        vlabel.configure(image=imtk_i)
        vlabel.photo = imtk_i
        i = len(level1_g)-1
        while i >= 0:
            x,y = level1_g[i]
            draw_i.ellipse((x-5,y-5,x+5,y+5),fill='white')
            i = i-1
        img_select1.save("/Users/hsiehyichin/Desktop/project/show/select1.bmp")
        img_select = Image.open("/Users/hsiehyichin/Desktop/project/show/select1.bmp")
        tkselect = ImageTk.PhotoImage(img_select)
        vlabel.configure(image=tkselect)
        vlabel.photo = tkselect
        var4.set(len(level1_g))
    #not hit
    else:
        on_hit = False
    return drawpoint

def l2():
    global on_hit
    global drawpoint
    #hit
    if on_hit == False:
        on_hit = True
        drawpoint = level2_g
        img_select2 = img_input.copy()
        imtk_i = ImageTk.PhotoImage(img_select2)
        draw_i= ImageDraw.Draw(img_select2)
        vlabel.configure(image=imtk_i)
        vlabel.photo = imtk_i
        i = len(level2_g)-1
        while i >= 0:
            x,y = level2_g[i]
            draw_i.ellipse((x-5,y-5,x+5,y+5),fill='white')
            i = i-1
        img_select2.save("/Users/hsiehyichin/Desktop/project/show/select2.bmp")
        img_select = Image.open("/Users/hsiehyichin/Desktop/project/show/select2.bmp")
        tkselect = ImageTk.PhotoImage(img_select)
        vlabel.configure(image=tkselect)
        vlabel.photo = tkselect
        var4.set(len(level2_g))
    #not hit
    else:
        on_hit = False
    return drawpoint

def l3():
    global on_hit
    global drawpoint
    #hit
    if on_hit == False:
        on_hit = True
        drawpoint = level3_g
        img_select3 = img_input.copy()
        imtk_i = ImageTk.PhotoImage(img_select3)
        draw_i= ImageDraw.Draw(img_select3)
        vlabel.configure(image=imtk_i)
        vlabel.photo = imtk_i
        i = len(level3_g)-1
        while i >= 0:
            x,y = level3_g[i]
            draw_i.ellipse((x-5,y-5,x+5,y+5),fill='white')
            i = i-1
        img_select3.save("/Users/hsiehyichin/Desktop/project/show/select3.bmp")
        img_select = Image.open("/Users/hsiehyichin/Desktop/project/show/select3.bmp")
        tkselect = ImageTk.PhotoImage(img_select)
        vlabel.configure(image=tkselect)
        vlabel.photo = tkselect
        var4.set(len(level3_g))
    #not hit
    else:
        on_hit = False
    return drawpoint

def scalebar():
    image_recover = Image.open("/Users/hsiehyichin/Desktop/project/show/input.bmp")
    emptyImage = image_recover.copy()
    imtk_r = ImageTk.PhotoImage(emptyImage)
    draw_r = ImageDraw.Draw(emptyImage)
    vlabel.configure(image=imtk_r)
    vlabel.photo = imtk_r
    i = len(drawpoint)-1
    s = var.get()
    if s == 0:
        var3.set("標示")
        while i >= 0:
            x,y = drawpoint[i]
            draw_r.ellipse((x-5,y-5,x+5,y+5),fill=(255,0,0))
            i = i-1
    elif s == 1:
        var3.set("傷口")
        while i >= 0:
            x,y = drawpoint[i]
            draw_r.ellipse((x-5,y-5,x+5,y+5),fill=(77,0,0))
            i = i-1
    elif s == 2:
        var3.set("結痂")
        while i >= 0:
            x,y = drawpoint[i]
            draw_r.ellipse((x-5,y-5,x+5,y+5),fill=(0,0,0))
            i = i-1
    elif s == 3:
        var3.set("癒合")
        while i >= 0:
            x,y = drawpoint[i]
            r1,g1,b1 = image_recover.getpixel((x,y+5))
            r2,g2,b2 = image_recover.getpixel((x,y-5))
            r3,g3,b3 = image_recover.getpixel((x+5,y))
            r4,g4,b4 = image_recover.getpixel((x-5,y))
            r = int((r1+r2+r3+r4)/4)+5
            b = int((b1+b2+b3+b4)/4)+5
            g = int((g1+g2+g3+g4)/4)+5
            draw_r.ellipse((x-5,y-5,x+5,y+5),fill=(r,g,b))
            draw_r.ellipse((x-3,y-3,x+3,y+3),fill=(0,0,0))
            i = i-1
    elif s == 4:
        var3.set("癒合")
        while i >= 0:
            x,y = drawpoint[i]
            r1,g1,b1 = image_recover.getpixel((x,y+5))
            r2,g2,b2 = image_recover.getpixel((x,y-5))
            r3,g3,b3 = image_recover.getpixel((x+5,y))
            r4,g4,b4 = image_recover.getpixel((x-5,y))
            r = int((r1+r2+r3+r4)/4)+5
            b = int((b1+b2+b3+b4)/4)+5
            g = int((g1+g2+g3+g4)/4)+5
            draw_r.ellipse((x-7,y-7,x+7,y+7),fill=(r,g,b))
            draw_r.ellipse((x-1,y-1,x+1,y+1),fill=(0,0,0))
            i = i-1
    elif s == 5:
        var3.set("康復")
        while i >= 0:
            x,y = drawpoint[i]
            r1,g1,b1 = image_recover.getpixel((x,y+5))
            r2,g2,b2 = image_recover.getpixel((x,y-5))
            r3,g3,b3 = image_recover.getpixel((x+5,y))
            r4,g4,b4 = image_recover.getpixel((x-5,y))
            r = int((r1+r2+r3+r4)/4)+5
            b = int((b1+b2+b3+b4)/4)+5
            g = int((g1+g2+g3+g4)/4)+5
            draw_r.ellipse((x-12,y-12,x+12,y+12),fill=(r,g,b))
            i = i-1
    emptyImage.save("/Users/hsiehyichin/Desktop/project/show/recover.bmp")
    emptyImage = Image.open("/Users/hsiehyichin/Desktop/project/show/recover.bmp")
    imtk_r = ImageTk.PhotoImage(emptyImage)
    vlabel.configure(image=imtk_r)
    vlabel.photo = imtk_r

#image
output = Image.open("/Users/hsiehyichin/Desktop/project/result/data11.bmp")
imtk = ImageTk.PhotoImage(output)
output_grab = Image.open("/Users/hsiehyichin/Desktop/project/show/show.bmp")
imtk_g = ImageTk.PhotoImage(output_grab)
width,height = output.size
w = int((640-width)/2)
vlabel=tk.Label(root1,image=imtk)
vlabel.place(x=600+w,y=120,anchor='nw')
#label
var2 = tk.StringVar()
var2.set("電腦的模擬")
tk.Label(root1,textvariable=var2,background="#F28482",foreground="#F7EDE2",font=("Helvetica",20)).place(x=870,y=665,anchor='nw')
#button
b1 = tk.Button(root1,text="電腦模擬",fg="#847577",font="helvetica",width=8,height=2,command=showresult_c)
b1.place(x=w+width+80,y=400,anchor='nw')
b2 = tk.Button(root1,text="手動選取",fg="#847577",font="helvetica",width=8,height=2,command=showresult_g)
b2.place(x=w+width+80,y=450,anchor='nw')
levelb1 = tk.Button(root1,text="1",font="helvetica",width=4,height=2,command=l1)
levelb1.place(x=1200,y=320,anchor='nw')
levelb2 = tk.Button(root1,text="2",font="helvetica",width=4,height=2,command=l2)
levelb2.place(x=1200,y=370,anchor='nw')
levelb3 = tk.Button(root1,text="3",font="helvetica",width=4,height=2,command=l3)
levelb3.place(x=1200,y=420,anchor='nw')
#scale
var = tk.IntVar()
s1 = tk.Scale(root1,from_=0,to=5,orient=tk.HORIZONTAL,length=100,showvalue=1,variable=var)
s1.place(x=1160,y=580,anchor='nw')
button = tk.Button(root1,text="recover",command=scalebar)
button.place(x=1190,y=630,anchor='nw')
#recover scale
var3 = tk.StringVar()
var3.set("標示")
info = tk.Label(root1,textvariable=var3,background="#E9F4D9",foreground="#569056",font=("Helvetica",20)).place(x=1190,y=550,anchor='nw')

#start the GUI
root1.mainloop()
