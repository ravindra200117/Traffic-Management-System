import cv2
import cv2.cv2 as cv2
import numpy as np
from datetime import datetime
import time as t

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from decimal import Decimal as D
def click(event,x,y,param,flag):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)


vidcap = cv2.VideoCapture("signal_video1.mp4")

BS_KNN = cv2.createBackgroundSubtractorKNN()
BS_MDG2 = cv2.createBackgroundSubtractorMOG2()

default_time = 10
constant_time = 5
# def calculate_percentage():
#     number_of_white_pix = np.sum(roi == 255)
#
#     number_of_black_pix = np.sum(roi == 0)
#     total = number_of_black_pix + number_of_white_pix
#     percentage_of_area_aquire = number_of_white_pix / total
#     result = percentage_of_area_aquire * 100
#
#     return result

def timing_loop(new_time):
    for i in range(new_time):
        print(i+1)
        t.sleep(1)

def predict_time(area):
    data = pd.read_csv('vehical_time.csv')
    x = data['percentage']
    y = data['time']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    arr = np.array(x_train)
    x_train_arr = arr.reshape(-1, 1)
    arr = np.array(y_train)

    y_train_arr = arr.reshape(-1, 1)
    arr = np.array(x_test)

    x_test_arr = arr.reshape(-1, 1)
    arr = np.array(y_test)

    y_test_arr = arr.reshape(-1, 1)
    reg = LinearRegression()

    reg.fit(x_train_arr, y_train_arr)

    # Prediction of Test and Training set result
    y_pred = reg.predict(x_test_arr)
    x_pred = reg.predict(x_train_arr)

    # predict the distance
    default_time = constant_time+int(reg.predict([[area]]))
    print(f"green signal --> {default_time} seconds")
    timing_loop(default_time)


img = cv2.imread("initial_image.png")

while(vidcap.isOpened()):
    ret,frame = vidcap.read()

    # now = datetime.now()
    # time = now.strftime("%H:%M:%S")



    # cv2.putText(frame, "Total vehicals:{}".format(vehicle), (450, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)
    # cv2.putText(frame, f"Time : {str(time)}", (450, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2)


    # cv2.setMouseCallback("original",click)
    if datetime.now().second == default_time:
        img = cv2.resize(img,(700,700))
        frame = cv2.resize(frame,(700,700))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_and(img,frame)
        thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,11)
        # cv2.imshow("substracted image",)
        cv2.imshow("Threshold image",thresh)
        cv2.waitKey(0)


    cv2.imshow("frames",frame)
    if cv2.waitKey(100) and 0xFF == 27:
        break


vidcap.release()
cv2.destroyAllWindows()