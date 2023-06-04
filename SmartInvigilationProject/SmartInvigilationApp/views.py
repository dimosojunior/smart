from django.db.models.query import QuerySet
from django.http.response import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, redirect, reverse, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User, auth
from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *


# project 2
from facenet_pytorch import MTCNN
from PIL import Image
from matplotlib import pyplot  as plt
import numpy as np
import math
import requests

import argparse
import torch
import cv2
#from .face import detect_faces
from . import NameFind
# import predFacePoseApp
# Create your views here.

def homePage(request):

	return render(request, 'SmartInvigilationApp/homePage.html')

def signin(request):

    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = auth.authenticate(username=username, password=password)

        if user is not None:
            auth.login(request, user)
            return redirect('homePage')
        else:
            messages.info(request, 'Credentials Invalid, Username or Password is incorrect')
            return redirect('signin')

    else:
        return render(request, 'SmartInvigilationApp/homePage.html')


def logout(request):
    auth.logout(request)
    return redirect('homePage')



















def project2(request):
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #C:\Users\DIMOSO JR\Desktop\ProjectWork\SmartInvigilation\SmartInvigilationProject\SmartInvigilationApp
    print(BASE_DIR)

# CODES ZA KUGET USERNAME AND PASSWORD
    username = request.POST.get('username')
    camera_no = request.POST.get('camera_no')
    print(username)
    print(camera_no)

    form = InvigilationStaffsForm()
    if request.method == 'POST':
        form = InvigilationStaffsForm(request.POST)
        if form.is_valid():
            form.save()

    
            left_offset = 20
            fontScale = 2
            fontThickness = 3
            text_color = (0,0,255)
            lineColor = (255, 255, 0)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f'Running on device: {device}')

            mtcnn = MTCNN(image_size=160,
                          margin=0,
                          min_face_size=20,
                          thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
                          factor=0.709,
                          post_process=True,
                          device=device # If you don't have GPU
                    )

            # Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
            def npAngle(a, b, c):
                ba = a - b
                bc = c - b 
                
                cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                
                return np.degrees(angle)

            # def visualize(image, landmarks_, angle_R_, angle_L_, pred_):
            #     fig , ax = plt.subplots(1, 1, figsize= (8,8))
                
            #     leftCount = len([i for i in pred_ if i == 'L'])
            #     rightCount = len([i for i in pred_ if i == 'R'])
            #     frontalCount = len([i for i in pred_ if i == ''])
            #     facesCount = len(pred_) # Number of detected faces (above the threshold)
            #     ax.set_title(f"Number of detected faces = {facesCount} \n frontal = {frontalCount}, left = {leftCount}, right = {rightCount}")
            #     for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
                    
            #         if pred == 'C':
            #             color = 'red'
            #         elif pred == 'R':
            #             color = 'blue'
            #         else:
            #             color = 'green'
                        
            #         point1 = [landmarks[0][0], landmarks[1][0]]
            #         point2 = [landmarks[0][1], landmarks[1][1]]

            #         point3 = [landmarks[2][0], landmarks[0][0]]
            #         point4 = [landmarks[2][1], landmarks[0][1]]

            #         point5 = [landmarks[2][0], landmarks[1][0]]
            #         point6 = [landmarks[2][1], landmarks[1][1]]
            #         for land in landmarks:
            #             pass
            #         #TO PRINT TRIANGLE AND CIRCLES ON A FACE
            #             #ax.scatter(land[0], land[1])
            #         # plt.plot(point1, point2, 'y', linewidth=3)
            #         # plt.plot(point3, point4, 'y', linewidth=3)
            #         # plt.plot(point5, point6, 'y', linewidth=3)
            #         #looking_center = int(pred)
            #         looking_right = int(math.floor(angle_R))
            #         looking_left = int(math.floor(angle_L))

            #         plt.text(point1[0], point2[0], f"{pred} \n {looking_left}, {looking_right}", 
            #                 size=20, ha="center", va="center", color=color)
            #         ax.imshow(image)
            #         fig.savefig(BASE_DIR+'/OutputImages/Output_detection.jpg')
            #     return print('Done detect')

            def visualizeCV2(frame, landmarks_, angle_R_, angle_L_, pred_):
                
                for landmarks, angle_R, angle_L, pred in zip(landmarks_, angle_R_, angle_L_, pred_):
                    
                    if pred == 'C':
                        color = (0, 255, 0) #Green-BGR
                    elif pred == 'Right Profile':
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                        
                    point1 = [int(landmarks[0][0]), int(landmarks[1][0])]
                    point2 = [int(landmarks[0][1]), int(landmarks[1][1])]

                    point3 = [int(landmarks[2][0]), int(landmarks[0][0])]
                    point4 = [int(landmarks[2][1]), int(landmarks[0][1])]

                    point5 = [int(landmarks[2][0]), int(landmarks[1][0])]
                    point6 = [int(landmarks[2][1]), int(landmarks[1][1])]

                    for land in landmarks:
                        pass
                        #UKITAKA KUWEKA LINE KWENYE FACE UNCOMMENT BELOW THEN TOA PASS HAPO JUU
                        # cv2.circle(frame, (int(land[0]), int(land[1])), radius=5, color=(0, 255, 255), thickness=-1)

                        
                    # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
                    # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                    # cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                    
                    text_sizeR, _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_PLAIN, fontScale, 4)
                    text_wR, text_hR = text_sizeR
                    
                    cv2.putText(frame, pred,(point1[0], point2[0]), cv2.FONT_HERSHEY_PLAIN, fontScale, color, fontThickness, cv2.LINE_AA)




            def predFacePose(frame):
                
                bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True) # The detection part producing bounding box, probability of the detected face, and the facial landmarks
                angle_R_List = []
                angle_L_List = []
                predLabelList = []

                x,y,width,height = 100,100,200,150
                color = (0,255,0)
                thickness = 2

                
                
                                

                
                
                if bbox_ is not None and prob_ is not None and landmarks_ is not None:
               
                    for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
                        if bbox is not None: # To check if we detect a face in the image
                            if prob > 0.9:#0.9 # To check if the detected face has probability more than 90%, to avoid 
                                angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
                                angL = npAngle(landmarks[1], landmarks[0], landmarks[2])# Calculate the left eye angle
                                angle_R_List.append(angR)
                                angle_L_List.append(angL)


                                            

                                            

                                if ((int(angR) in range(35, 57)) and (int(angL) in range(35, 58))):
                                    predLabel='C' #'Frontal'
                                    predLabelList.append(predLabel)
                                else: 
                                    if angR < angL:
                                        LeftAngle = int(angL)
                                        predLabel= f'{LeftAngle}-R'

                                        print(f"angR (θ2) = {angR} and angL (θ1) =  {angL}, Hence, Student is Looking Right at Angle {angL} ")
                                        print(" ")
                                        print(" ")
                                        

                                        if angL > 80: #-80
                                            cv2.circle(frame, (int(landmarks[1][0]), int(landmarks[1][1])), radius=50, color=(0, 0, 255), thickness=5)
                                            predLabel= f'{LeftAngle}' #'Cheating L' 

                                            print("STUDENT'S FACE IS OUTSIDE THE BOUNDARY, THEN; ")
                                            print(" ")
                                            

                                            print(f"Student is Cheating Right at Angle {angL} ")
                                            print(" ")
                                            print(" ")
                                            

                                            # KWA AJILI YA KUCHORA PEMBE TATU
                                            # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
                                            # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                                            # cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                                                                                        
                                    else:
                                        RightAngle = int(angR)
                                        predLabel=f'{RightAngle}-L'



                                        print(f"angR (θ2) = {angR} and angL (θ1) =  {angL}, Hence, Student is Looking Left at Angle {angR} ")
                                        print(" ")
                                        

                                        if angR > 80: #80
                                            cv2.circle(frame, (int(landmarks[0][0]), int(landmarks[1][1])), radius=50, color=(0, 0, 255), thickness=5)
                                            predLabel=f'{RightAngle}' #'Cheating R'

                                            print("STUDENT'S FACE IS OUTSIDE THE BOUNDARY, THEN; ")
                                            print(" ")
                                            


                                            print(f"Student is Cheating Left at Angle {angR} ")
                                            print(" ")
                                            print(" ")
                                            

                                            # KWA AJILI YA KUCHORA PEMBE TATU
                                            # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[1][0]), int(landmarks[1][1])), lineColor, 3)
                                            # cv2.line(frame, (int(landmarks[0][0]), int(landmarks[0][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                                            # cv2.line(frame, (int(landmarks[1][0]), int(landmarks[1][1])), (int(landmarks[2][0]), int(landmarks[2][1])), lineColor, 3)
                                            
                                            #cv2.rectangle(frame,(x,y),(x + width, y + height), color, thickness)


                                    predLabelList.append(predLabel)
                            else:
                                print('The detected face is Less then the detection threshold')
                                continue
                        else:
                            print('No face detected in the image')
                            continue
                    # FACE YA MTU INAONEKANA KWENYE CAMERA
                    # VALUES ZAKE NDO HIZI KWA ANGLES ZOTE

                    # print(f"right {angle_R_List} ")
                    # print(f"left {angle_L_List} ")
                    # print(f"center {predLabelList} ")
                    
                    return landmarks_, angle_R_List, angle_L_List, predLabelList
                else:
                    # KAMAHAMNA MTU KWENYE CAMERA IKASOME HIZI DEFAULT
                    # VALUES ILI KUREMOVE ERROR INAYOSEMA "CAN NOT UNPACK NONETYPE OBJECT"
                    angle_R_List = [41.499546, 38.9971]
                    angle_L_List= [44.377758, 45.907673]
                    predLabelList= ['', '']
                    # bbox_ = [[-76.825165 345.9478    87.617516 547.3037  ]
                    #          [302.04672  163.34067  380.0983   259.25977 ]
                    #          [120.40265  130.62308  186.46005  210.27997 ]]
                    landmarks_ = [[[-34.039093,   421.1461    ],
                                  [ 16.470798,   427.12183   ],
                                  [-35.07744,    464.84344   ],
                                  [-41.21428,    498.56042   ],
                                  [ -0.88659286, 504.5458    ]],

                                 [[324.0439,     198.46864   ],
                                  [360.09473,    200.7623    ],
                                  [339.04395,    216.24658   ],
                                  [324.7361,     238.5289    ],
                                  [353.1474,     240.4968    ]],

                                 [[133.74072,    161.84549   ],
                                  [163.7682,     158.71199   ],
                                  [145.17691,    176.29892   ],
                                  [138.92558,    193.42422   ],
                                  [164.13974,    191.53346   ]]]
                    # prob_ =[0.9998667, 0.9999882, 0.9963965]
                    #return landmarks_, angle_R_List, angle_L_List, predLabelList
                    print("HAKUNA FACE YA MTU INAYOONEKANA KWENYE CAMERA")
                    # print("None")
                    # print(f"r is {angle_R_List} ")
                    # print(f"l is {angle_L_List} ")
                    # print(f"prob_ is {predLabelList} ")

                    return landmarks_, angle_R_List, angle_L_List, predLabelList
                

            def predFacePoseApp():

                source = 0

                #mysource = camera_no #now naingiza video path lkn km unaingiza no?
                mysource = int(camera_no)
                #print(camera_no)
                # Create a video capture object from the VideoCapture Class.
                #video_cap = cv2.VideoCapture(BASE_DIR+"/videos/4.mp4")
                #video_cap = cv2.VideoCapture(BASE_DIR+mysource)

                #km unaingiza no path itakuwa
                video_cap = cv2.VideoCapture(mysource)

                # Create a named window for the video display.
                win_name = 'SMART INVIGILATION SYSTEM'
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                # video_cadesired_width = 1400
                # desired_height = 800
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(win_name, 1400, 1000)
                #dim = (video_cadesired_width, desired_height)
                left_offset = 20
                fontScale = 2
                fontThickness = 3
                text_color = (255,0,0) #(0,0,255)

                # #Mwisho wa full screen Model
                


                #MWANZO  KWA AJILI YA KURECODI VIDEO

                #video_cap = cv2.VideoCapture(BASE_DIR+"/videos/6.mp4")
                fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
                capturing_win_name = 'CAPTURING LIVE VIDEO'
                cv2.namedWindow(capturing_win_name, cv2.WINDOW_NORMAL)
                # video_cadesired_width = 1400
                # desired_height = 800
                cv2.setWindowProperty(capturing_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.resizeWindow(capturing_win_name, 1400, 1000)
                # out=cv2.VideoWriter(BASE_DIR+'/saved-media/my.mp4',fourcc,20.0,(640,480))

                if (video_cap.isOpened() == False):
                    print("Unable to read camera")

                frame_width = int(video_cap.get(3))
                frame_height = int(video_cap.get(4))
                out=cv2.VideoWriter(BASE_DIR+'/saved-media/Smart Invigilation Video.avi',fourcc,10,(frame_width,frame_height))


                #MWISHO  KWA AJILI YA KURECODI VIDEO





                
                
                face_cascade = cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier(BASE_DIR+'/cascades/data/haarcascade_eye.xml')
                







                while True:

                    






                    #MWANZO  KWA AJILI YA KURECODI VIDEO

                    ret, frame = video_cap.read()
                    if (ret==True):
                        cv2.flip(frame,180)
                        out.write(frame)

                    # Display the resulting frame
                        cv2.imshow(capturing_win_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        continue

                    #MWISHO  KWA AJILI YA KURECODI VIDEO

                


                    # Read one frame at a time using the video capture object.
                    has_frame, frame = video_cap.read()
                    if not has_frame:
                        #break
                        print("imeshindwa kuread image kutoka kwenye camera")
                        continue



                    



                    #DRAWING BOX ON STUDENT'S FACE


                    faces = face_cascade.detectMultiScale(frame, 1.3, 5)             
                    for (x, y, w, h) in faces:                                         
                        NameFind.draw_box(frame, x, y, w, h)

                    #cv2.imshow('Face Detection Using Haar-Cascades ', frame)         
                    if cv2.waitKey(1) & 0xFF == ord('q'):                           
                        break


                    #MWISHO WA  DRAWING BOX 



                    landmarks_, angle_R_List, angle_L_List, predLabelList = predFacePose(frame)

                    # Annotate each video frame.
                    visualizeCV2(frame, landmarks_, angle_R_List, angle_L_List, predLabelList)

                    #To draw a graphy

                    #visualize(frame, landmarks_, angle_R_List, angle_L_List, predLabelList)

                    cv2.imshow(win_name, frame)

                    key = cv2.waitKey(1)

                    # You can use this feature to check if the user selected the `q` key to quit the video stream.
                    if key == ord('Q') or key == ord('q') or key == 27:
                        # Exit the loop.
                        break

                video_cap.release()
                cv2.destroyWindow(win_name)
                
            

            messages.success(request, f"Invigilation Completed Successfully By - {username} ")
            predFacePoseApp()


            return redirect('starting_page')
                #return HttpResponse("welll")
                #return render(request, 'SmartInvigilationApp/homePage.html')
    

@login_required(login_url='homePage')
def starting_page(request):

    return render(request, 'SmartInvigilationApp/starting_page.html')



def record_video(request):
    import numpy as np
    import cv2
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    win_name = 'SMART INVIGILATION SYSTEM'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # video_cadesired_width = 1400
    # desired_height = 800
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(win_name, 1400, 1000)

    username = request.user

    
    mycamera = cv2.VideoCapture(BASE_DIR+"/videos/3.mp4")
    fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    # out=cv2.VideoWriter(BASE_DIR+'/saved-media/my.mp4',fourcc,20.0,(640,480))

    if (mycamera.isOpened() == False):
        print("Unable to read camera")

    frame_width = int(mycamera.get(3))
    frame_height = int(mycamera.get(4))
    out=cv2.VideoWriter(BASE_DIR+'/saved-media/video.avi',fourcc,10,(frame_width,frame_height))


    while(True):
        # mycamerature frame-by-frame
        ret, frame = mycamera.read()
        if (ret==True):
            cv2.flip(frame,180)
            out.write(frame)

        # Display the resulting frame
            cv2.imshow(win_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the mycamerature
    mycamera.release()
    cv2.destroyAllWindows()
    messages.success(request, f"Video Recorded Successfully By {username} ")
    return redirect('homePage')



    
