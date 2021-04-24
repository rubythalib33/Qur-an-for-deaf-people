import cv2
import numpy as np
import os
import mediapipe as mp
 
def resize_image(scale_percent, img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

data_path = './Dataset/ArASL_Database_54K_Final/'
result_path = './Dataset/extracted/'

listdir = os.listdir(result_path)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode = True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

for foldername in listdir:
    print('extracting', foldername)
    folder_path = data_path+foldername
    listfile = os.listdir(folder_path)
    for filename in listfile:
        print('try', filename)
        file_ = folder_path+'/'+filename
        
        image = cv2.imread(file_)
        image = resize_image(400,image)
        #preprocess basic for image
        
        result_hand = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result_coor = []

        if result_hand.multi_hand_landmarks:
            for hand_landmarks in result_hand.multi_hand_landmarks:
                for data_point in hand_landmarks.landmark:
                    lm = [data_point.x, data_point.y, data_point.z]
                    result_coor.append(lm)
                mp_drawing.draw_landmarks(
                    image=image,landmark_list= hand_landmarks,connections=mp_hands.HAND_CONNECTIONS
                )
            
            file1 = open(result_path+foldername+'/'+filename+'.txt',"w")
            file1.write(str(result_coor))
            file1.close()

            cv2.imwrite(result_path+foldername+'/'+filename, image)