import os
import tkinter
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from tkinter import messagebox

# select device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
name = input("Enter a name: ")

def get_face(name, num):
    if not os.path.exists(f'dataset/{name}'):
        os.makedirs(f'dataset/{name}')
    else:
        print("Name existed!! Enter another name!")
        name = input("Enter a name: ")
        os.makedirs(f'dataset/{name}')

    # mtcnn detect face position
    mtcnn = MTCNN(device=device, keep_all=True)

    # Initialize the video window
    windows_name = 'face'
    cv2.namedWindow(windows_name)
    cap = cv2.VideoCapture(0)

    count = 0
    while True:
        # Read an image from the camera
        success, image = cap.read()
        if not success:
            break

        # detect face position
        boxes, probs = mtcnn.detect(image)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # Set face detection threshold
                if prob < 0.9:
                    continue

                x1, y1, x2, y2 = [int(p) for p in box]

                # Save the current face as a picture
                face_image_name = f'dataset/{name}/{name}_{count}.jpg'
                count += 1
                if count > num:
                    break
                print(face_image_name)
                face_image = image[y1 - 10:y2 + 10, x1 - 10: x2 + 10]
                cv2.imwrite(face_image_name, face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                # Frame the face position
                cv2.rectangle(image, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), color=(0, 255, 0), thickness=1)
                cv2.putText(image, f'Shooting:{round(count/100*40)}''%', (x1, y1 - 30), cv2.FONT_ITALIC, 1, (51, 102, 255), 4)
            
        # Display the processed image
        cv2.imshow(windows_name, image)

        if count > num:
            break
        # keep the window
        key = cv2.waitKey(1)
        if key & 0xff == 27:
            break

    # Release device resources and destroy the window
    cap.release()
    cv2.destroyAllWindows()
    print("Finished shooting")   
    os.system('python model_training.py')

if __name__ == '__main__':
    get_face(name, 250)
