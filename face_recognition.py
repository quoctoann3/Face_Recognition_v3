'''
TOANNQ 2022

'''

from calendar import day_abbr
from contextlib import nullcontext
import encodings
from importlib.resources import path
from itertools import count
from operator import le
from tkinter import Frame
from unicodedata import name
import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import time
import os.path
import csv
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

timestr = time.strftime("%Y%m%d-%H%M%S")

#CSV
f = open('attendance.csv', 'a', encoding='UTF8',newline='')
writer = csv.writer(f)

# face recognizer
class FaceRecognizer:
    # Initialize, load data
    def __init__(self, knn_model_path='knn_model.pkl', face_feature_path='face_feature.csv'):
        # select device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))

        # Read the trained face feature data
        self.data = pd.read_csv(face_feature_path)
        self.x = self.data.drop(columns=['label'])
        self.y = self.data['label']

        # Load the trained KNN classifier model
        self.knn_model = joblib.load(knn_model_path)

        # font
        self.font = ImageFont.truetype('simsun.ttc', size=30)

    # Recognize faces according to feature vectors, use Euclidean distance, if the distance is greater than 1, the recognition is considered failed
    # The function of the KNN model is repeated here, but I just want to calculate a minimum distance, which slightly affects the recognition performance
    def _recognize(self, v):
        dis = np.sqrt(sum((v[0] - self.x.iloc[0]) ** 2))
        name = self.y[0]

        for i in range(1, self.x.shape[0]):
            temp_dis = np.sqrt(sum((v[0] - self.x.iloc[i]) ** 2))
            if temp_dis < dis:
                dis = temp_dis
                name = self.y[i]

        return name, dis
    # face recognition main function
    def start_recognize(self):
        # mtcnn detect face position
        mtcnn = MTCNN(device=self.device, keep_all=True)
        # Used to generate face 512-dimensional feature vector
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Initialize the video window
        windows_name = 'FACE RECOGNITION'
        cv2.namedWindow(windows_name)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS,30) 
        current_frame = 0

        while True:
            # Read an image from the camera
            success, image = cap.read()
            current_frame += 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(image,  (0,30), ( 1280+120,-30+-80), (0,200,0), -1)
            cv2.putText(image,time.strftime("%Y/%m/%d-%H:%M:%S"), (5,20),font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(image,'frame:'+ str(current_frame), (1150,700),font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image,'FPS:'+ str(cap.get(cv2.CAP_PROP_FPS)), (1150,700),font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

            if not success:
                break

            img_PIL = Image.fromarray(image)
            draw = ImageDraw.Draw(img_PIL)

            # Detect face position, get face frame coordinates and face probability
            boxes, probs = mtcnn.detect(image)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    # Set face recognition threshold
                    if prob < 0.9:
                        continue
                    x1, y1, x2, y2 = [int(p) for p in box]
                    # Frame the face position
                    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                    # cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color=(0, 255, 0), thickness=2)
                    # cv2.putText(image, str(round(prob, 3)), (x1, y1 - 30), cv2.FONT_ITALIC, 1, (255, 0, 255), 4)

                    # export face image
                    face = mtcnn.extract(image, [box], None).to(self.device)
                    # Generate 512-dimensional feature vector
                    embeddings = resnet(face).detach().cpu().numpy()
                    # KNN prediction
                    name_knn = self.knn_model.predict(embeddings)

                    # get predicted name and distance
                    _, dis = self._recognize(embeddings)
                    # If the distance is too large, the recognition fails
                    if dis > 0.9:
                        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 255), width=4)
                        cv2.rectangle(image,  (0,30), ( 1280+120,-30+-80), (0,0,255), -1)
                        cv2.putText(image,time.strftime("%Y/%m/%d-%H:%M:%S")+"    [Unknown]", (5,20),font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                        draw.text((x1, y1 - 40), f'Unknown!', font=self.font, fill=(0, 0, 255))
                        path = os.path.sep.join(['unknown', "{}.jpg".format(time.strftime("%Y%m%d-%H%M%S")+"unknown")])
                        cv2.imwrite(path,image)
                    else:
                        # Frame the face position and write the name
                        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
                        draw.text((x1, y1 - 40), f'{name_knn[0]}({round(float(prob*100),2)})', font=self.font, fill=(0, 255, 0))                        
                        print(name_knn,{round(float(prob*100),2)},time.strftime("%Y/%m/%d"),time.strftime("%H:%M:%S"))
                        cv2.putText(image,str(name_knn[:]), (350,20),font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                        path = os.path.sep.join(['attendance', "{}.jpg".format(time.strftime("%Y%m%d-%H%M%S")+"-"+str(name_knn[0]))])
                        cv2.imwrite(path,image)
                        data = [time.strftime("%Y/%m/%d"),time.strftime("%H:%M:%S"), name_knn[0]]
                        #CSV write the data
                        writer.writerow(data)

            # Display the processed image
            cv2.imshow(windows_name, np.array(img_PIL))

            # keep the window
            key = cv2.waitKey(1)
         
            # ESC key to exit
            if key & 0xff == 27:
                f.close()
                break

        # Release device resources and destroy the window
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        fr = FaceRecognizer()
        fr.start_recognize()
    except Exception as e:
        print(e)
