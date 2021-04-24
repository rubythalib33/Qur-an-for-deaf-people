import cv2
# import mediapipe as mp
import numpy as np
from util import *
from PIL import ImageFont, ImageDraw, Image  

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands
model_qtn = load_model()


# hands = mp_hands.Hands(
#     min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

class_names = ['ain', 'al', 'aleff', 'bb', 'dal', 'dha', 'dhad', 'fa', 'gaaf', 'ghain', 'ha', 'haa', 'jeem', 'kaaf', 'khaa', 'la', 'laam', 'meem', 'nun', 'ra', 'saad', 'seen', 'sheen', 'ta', 'taa', 'thaa', 'thal', 'toot', 'waw', 'ya', 'yaa', 'zay']
ar = {'ain':'ع', 'al':'آل', 'aleff':'أ', 'bb':'ب', 'dal':'د', 'dha':'ظ', 'dhad':'ض','fa':'ف', 'gaaf':'ق','ghain':'غ', 'ha':'ه', 'haa':'ح', 'jeem':'ج', 'kaaf':'ك', 'khaa':'خ', 'la':'لا', 'laam':'ل', 'meem':'م', 'nun':'ن', 'ra':'ر', 'saad':'ص', 'seen':'س', 'sheen':'ش', 'ta':'ط', 'taa':'ت', 'thaa':'ث', 'thal':'ذ', 'toot':'ة', 'waw':'و', 'ya':'ي', 'yaa':'يا', 'zay':'ز'}
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    # results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_hight, image_width, _ = image.shape

    # cv2.rectangle(image, (340,50), (640,350), (255,0,0), 1)

    # if results.multi_hand_landmarks:
    #     for hand_landmark in results.multi_hand_landmarks:
    #         x = [landmark.x for landmark in hand_landmark.landmark]
    #         y = [landmark.y for landmark in hand_landmark.landmark]
            
    #         center = np.array([np.mean(x)*image_width, np.mean(y)*image_hight]).astype('int32')
    #         cv2.circle(image, tuple(center), 10, (255,0,0), 1)
    #         cv2.rectangle(image, (center[0]-200,center[1]-200), (center[0]+200,center[1]+200), (255,0,0), 1)

    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hand = greyscale[50:250, 390:590]
    cv2.rectangle(image, (390,50), (590,250), (255,0,0), 1)
    # print(hand.shape)
    result_qtn, prediction = recognition(model_qtn, hand)
    str_ = ar[class_names[result_qtn]]
    if prediction > 0.3:
        print(str_)
        # cv2.putText(image, str_, (50,50), font='./aset/Arabic Bold.ttf' ,  
        #            fontScale=1, color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('hand', hand)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
# hands.close()
cap.release()

#(x_min,y_min),(x_max,y_max) = (center[0]-200,center[1]-200), (center[0]+200,center[1]+200)
# cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (255,0,0), 1)

# greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# hand = np.zeros((y_max-y_min+1, x_max-x_min+1))