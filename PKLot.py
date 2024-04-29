import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('PKLot.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

areas = [
    [(89,357),(72,385),(112,385),(131,356)],
    [(136,355),(117,387),(153,385),(169,357)],
    [(175,356),(159,384),(194,387),(210,355)],
    [(213,356),(199,385),(233,385),(249,356)],
    [(254,357),(238,389),(279,386),(291,357)],
    [(294,357),(281,389),(322,388),(333,356)],
    [(337,356),(328,387),(365,390),(373,358)],
    [(379,356),(370,386),(409,389),(416,359)],
    [(422,359),(414,390),(452,391),(459,357)],
    [(463,358),(460,389),(497,391),(501,358)],
    [(505,358),(501,390),(539,391),(540,357)],
    [(548,360),(543,390),(583,389),(583,359)],
    [(587,360),(588,388),(631,393),(628,358)],
    [(633,358),(635,390),(676,391),(672,360)]
]


while True:    
    ret,frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)




    counts = [0,1,1,1,0,0,0,0,0,0,1,0,1,0]
    



    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2




            for i in range(len(counts)):
                result = cv2.pointPolygonTest(np.array(areas[i],np.int32),((cx,cy)),False)
                if result>=0:
                    counts[i] = 1              
            



    for i in range(len(counts)):
        if counts[i] == 1:
            cv2.polylines(frame,[np.array(areas[i],np.int32)],True,(0,0,255),2)
        else:
          cv2.polylines(frame,[np.array(areas[i],np.int32)],True,(0,255,0),2)  





    cv2.imshow("RGB", frame)

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
