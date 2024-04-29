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

cap=cv2.VideoCapture('parking.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

areas = [
    [(52,364),(30,417),(73,412),(88,369)],
    [(105,353),(86,428),(137,427),(146,358)],
    [(159,354),(150,427),(204,425),(203,353)],
    [(217,352),(219,422),(273,418),(261,347)],
    [(274,345),(286,417),(338,415),(321,345)],
    [(336,343),(357,410),(409,408),(382,340)],
    [(396,338),(426,404),(479,399),(439,334)],
    [(458,333),(494,397),(543,390),(495,330)],
    [(511,327),(557,388),(603,383),(549,324)],
    [(564,323),(615,381),(654,372),(596,315)],
    [(616,316),(666,369),(703,363),(642,312)],
    [(674,311),(730,360),(764,355),(707,308)]
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




    counts = [0,0,0,0,0,0,0,0,0,0,0,0]
    



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
