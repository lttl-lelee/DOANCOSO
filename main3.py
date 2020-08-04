import cv2
import numpy as np
import tkinter as tk
from tkinter import Message ,Text
import tkinter.ttk as ttk
import tkinter.font as font
import csv
import cv2,os
import shutil
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from train import *
from facenet.face_contrib import *
from align.align_mtcnn import *
import sqlite3
from datetime import date
from datetime import datetime
# import tensorflow as tf

#conn = sqlite3.connect("FaceBase.db")
# Giao dien
window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
window.geometry("900x620")
window.configure(background='white')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

background_image = tk.PhotoImage(file='logo.png')

background_label = tk.Label(window, image=background_image ,bg="white") 

background_label.place(x=20, y=10)

lbl = tk.Label(window, text="MSV",width=20  ,fg="black"  ,bg="#EEEEEE"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl.place(x=60, y=160)

txt = tk.Entry(window,width=20  ,textvariable=1,bg="white"  ,fg="black",font=('times', 15, ' bold '))
txt.place(x=320, y=160,width=300, height=50)

lbl2 = tk.Label(window, text="ENTER NAME",width=20  ,fg="black"  ,bg="#EEEEEE"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=60, y=240)

txt2 = tk.Entry(window,width=20  ,textvariable=2,bg="white"  ,fg="black",font=('times', 15, ' bold ')  )
txt2.place(x=320, y=240,width=300, height=50)

lbl3 = tk.Label(window, text="NOTIFICATION: ",width=20  ,fg="black"  ,bg="#EEEEEE"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=60, y=320)

message = tk.Label(window, text="" ,bg="white" , fg="red"  ,width=41  ,height=2, borderwidth=2, relief="groove",activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=320, y=320)

# lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold  underline')) 
# lbl3.place(x=400, y=700)


# message2 = tk.Label(window, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
# message2.place(x=700, y=700)

# Load HAAR face classifier, tập phân loại gương mặt
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False



# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face

def add_overlays(frame, faces, frame_rate, colors, confidence=0.4):
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            #cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (255,255,255), 2)
            if face.name and face.prob:
                if face.prob > confidence:
                    class_name = face.name
                    profile = getProfile(class_name)
                    if(str(profile[3] == 1)):
                        conn=sqlite3.connect("FaceBase4.db")
                        cmd = "UPDATE sinhvien SET DIEMDANH=? WHERE MASV=?"
                    else:
                        cmd="INSERT INTO sinhvien(DIEMDANH) Values(?) WHERE MASV=?"
                    cursor=conn.execute(cmd,[1,class_name])
                    conn.commit()
                    conn.close()
                else:
                    class_name = 'Unknow'
                    # class_name = face.name
                if(profile!=None):
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    conn=sqlite3.connect("FaceBase4.db")
                    cmd="INSERT INTO diemdanh(MASV,DATETIME,COMAT) Values(?,?,?) "
                    cursor=conn.execute(cmd,[class_name,dt_string,1])
                    conn.commit()
                    conn.close()
                    cv2.putText(frame, "MASV: " + str(profile[1]), (face_bb[0], face_bb[3] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
                    cv2.putText(frame, "NAME: " + str(profile[2]), (face_bb[0], face_bb[3] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255) ,2)
                else:
                    cv2.putText(frame, "Unknow", (face_bb[0], face_bb[3] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255) ,2)
                    #cv2.putText(img, "Gender: " + str(profile[3]), (x,y+h+90), fontface, fontscale, fontcolor ,2)
                # cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                #             colors[idx], thickness=2, lineType=2)
                # cv2.putText(frame, '{:.02f}'.format(face.prob * 100), (face_bb[0], face_bb[3] + 40),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=1, lineType=2)

    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #             thickness=2, lineType=2)

#insert/update data to sqlite
def insertOrUpdate(masv,name):
    conn=sqlite3.connect("FaceBase4.db")
    cmd = "SELECT * FROM sinhvien WHERE MASV=?"
    # cmd="SELECT * FROM sinhvien WHERE MASV="+str(masv)
    cursor=conn.execute(cmd, [masv])
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE sinhvien SET NAME=? WHERE MASV=?"
        res = "Mã sinh viên đã tồn tại, cập nhật thông tin thành công !"
        message.configure(text= res)
    else:
        cmd="INSERT INTO sinhvien(NAME,MASV) Values(?,?)"
    conn.execute(cmd, [name, masv])
    conn.commit()
    conn.close()

def TakeImages():        
    masv=(txt.get())
    name=(txt2.get())
    #if(Id and name.isalpha()):
    if(txt.get()=="" or txt2.get()==""):
        res = "Vui lòng nhập đủ thông tin !"
        message.configure(text= res)
    else:
        #ket noi database
        insertOrUpdate(masv,name)
        # Initialize Webcam
        path = "Datasets/" + masv
        if not os.path.exists(masv):
            os.mkdir(path)

        cap = cv2.VideoCapture(0)
        count = 0
        # Collect 100 samples of your face from webcam input
        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (400, 400))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # Save file in specified directory with unique name
                #file_name_path = 'Datasets/' +name + "/" + name +"."+Id +'.'+ str(count) + '.jpg'
                file_name_path = 'Datasets/' +masv + "/" + masv +"."+ str(count) + '.jpg'
                print(file_name_path)
                cv2.imwrite(file_name_path, face)

                # Put count on images and display live count
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Cropper', face)
                
            else:
                print("Face not found")
                pass

            if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
                break
                
        cap.release()
        cv2.destroyAllWindows()      
        print("Collecting Samples Complete")
        #res = "Images Saved for ID : " + Id +" Name : "+ name
        res = "Images Saved for Name : "+" Name : "+ masv
        #row = [Id , name]
        row = [name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)

def TrainImages():
    align_mtcnn('Datasets', 'face_align')
    train_data('face_align/', 'models/20180402-114759.pb', 'models/your_model.pkl')
    res = "Image Trained"
    message.configure(text= res)

def getProfile(masv):
    conn=sqlite3.connect("FaceBase4.db")
    cmd="SELECT * FROM sinhvien WHERE MASV=?"
    cursor=conn.execute(cmd,[masv])
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

def TrackImages():
    model_checkpoint, classifier = 'models', 'models/your_model.pkl'
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()
    width = frame.shape[1]
    height = frame.shape[0]
    
    face_recognition = Recognition(model_checkpoint, classifier)
    start_time = time.time()
    colors = np.random.uniform(0, 255, size=(1, 3))
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)
            for i in range(len(colors), len(faces)):
                colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate, colors)

        frame_count += 1
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   
    video_capture.release()
    cv2.destroyAllWindows()
    
clearButton = tk.Button(window, text="CLEAR", command=clear  , fg="black"  ,bg="white" ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton.place(x=630, y=170)
clearButton2 = tk.Button(window, text="CLEAR", command=clear2,  fg="black"  ,bg="white" ,width=15  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=630, y=250)    
takeImg = tk.Button(window, text="DATA COLLECTION", command=TakeImages , fg="black"  ,bg="#EEEEEE"  ,width=25  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=150, y=420)
trainImg = tk.Button(window, text="TRAIN DATA",command=TrainImages , fg="black"  ,bg="#EEEEEE"  ,width=25  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=470, y=420)
trackImg = tk.Button(window, text="FACE RECOGNITION" ,command=TrackImages,fg="black"  ,bg="#EEEEEE"  ,width=25  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=150, y=500)
quitWindow = tk.Button(window, text="QUIT",command=window.destroy ,fg="black"  ,bg="#EEEEEE"  ,width=25  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=470, y=500)

 
window.mainloop()