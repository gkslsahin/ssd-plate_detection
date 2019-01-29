import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="image path")
ap.add_argument("-p","--pbtxt",default="model/graph.pbtxt",help="pbtxt file")
ap.add_argument("-m","--model",default="model/frozen_inference_graph.pb",help="model file().pb")
args = vars(ap.parse_args())

LABELS = ["null","plate"]     

cvNet = cv.dnn.readNetFromTensorflow(args["model"], args["pbtxt"])

img = cv.imread(args["image"])
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.5:
        label = "{}: {:.2f}%".format(LABELS[int(detection[1])], detection[2] * 100)
        print("[INFO] {}".format(label))

        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        print("left:{} top:{} right:{} bottom:{}".format(left,top,right,bottom))
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
        left = left - 15 if left - 15 > 15 else left + 15
        top = top - 5
        cv.putText(img, label, (int(left), int(top)),
            cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)

cv.imshow('img', img)
cv.waitKey()