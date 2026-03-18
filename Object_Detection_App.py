import torch
import cv2
import numpy as np

model=torch.hub.load("ultralytics/yolov5","yolov5s",pretrained=True)

def detect_image(image1):
    image=cv2.imread(image1)
    results=model(image)
    results.show()

def detect_webcam():
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        results=model(frame)
        rendered_image=np.array(results.render()[0])
        cv2.imshow("webcam",rendered_image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("welcome to object detection app")
    print("1-detect objects in image")
    print("2-detect objects from webcam")
    choice=int(input("enter your choice:"))
    if choice==1:
        image=input("enter the image path")
        detect_image(image)
    elif choice==2:
        detect_webcam()
    else:
        print("enter valid choice")

if __name__=="__main__":
    main()
