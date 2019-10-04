from PIL import ImageDraw,Image
import cv2
import numpy as np
import math

image_rgb = Image.open("/Users/hsiehyichin/Desktop/freckle1.bmp")    #(r,g,b)
image_gray = image_rgb.convert("L")  #gray
draw = ImageDraw.Draw(image_rgb)     #drawable
width = image_gray.width 
height = image_gray.height
color_min = 255  #white
color_max = 0    #black
x = 0
y = 0
allcolor = []   #有重複的顏色集
mode = []       #眾數
mark_x = []
mark_y = []
mark = []
counter = 0

#counting how much colors show function
def counting(list):
    #color = []                                                    #沒有重複的顏色集
    list_appear = dict((a, allcolor.count(a)) for a in allcolor)   #總共出現幾次
    if max(list_appear.values()) == 1:                             #都只出現一次
        return                                                     #沒有眾數
    else:
        for k,v in list_appear.items():                            #k為key,v為value
            #color.append(k)
            if v == max(list_appear.values()):
                mode.append(k)                                     #收集顏色最深的點
    return mode

#make color list
while y < height:
    while x < width:
        allcolor.append(image_gray.getpixel((x,y)))
        x = x+1
    y = y+1
    x = 0  #initialize

counting(allcolor)
print(mode)

#make mark list
y = 0  #initialize
i = 0
while y < height:
    while x < width:
        point = image_gray.getpixel((x,y))
        while i < len(mode):
            if point < mode[i]:         ###adjust
                mark_x.append(x)
                mark_y.append(y)
                mark.append((x,y))
                counter = counter+1
            i = i+1
        x = x+1
        i = 0
    y = y+1
    x = 0

#mark
i = 0 #initialize
while i < counter:
    point_x = mark_x[i]
    point_y = mark_y[i]
    draw.ellipse((mark_x[i],mark_y[i],mark_x[i]+12,mark_y[i]+12),
                 fill=(255,255,255),outline ='blue',width=1)
    mark_x.remove(mark_x[i])
    mark_y.remove(mark_y[i])
    for scan_x in mark_x :
        if pow(point_x-scan_x,2) < 576:
            index = mark_x.index(scan_x)
            mark_x.remove(scan_x)
            mark_y.remove(mark_y[index])
            counter = counter-1   
    counter = counter -1



#draw.ellipse((x,y,x+12,y+12),fill=(255,255,255),outline ='blue',width=1)
image_rgb.save('/Users/hsiehyichin/Desktop/result2.bmp')



