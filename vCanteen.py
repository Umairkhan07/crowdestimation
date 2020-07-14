#from filetype import filetype
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import cv2
import time
import datetime
import numpy as np
import requests
from subprocess import Popen, PIPE
import re
MAX_COUNT = 240


def get_MCNN():
    input1 = Input(shape=(None, None, 1))

    # S
    xs = Conv2D(24, kernel_size=(5, 5), padding='same', activation='relu')(input1)
    xs = MaxPooling2D(pool_size=(2, 2))(xs)
    xs = Conv2D(48, kernel_size=(3, 3), padding='same', activation='relu')(xs)
    xs = MaxPooling2D(pool_size=(2, 2))(xs)
    xs = Conv2D(24, kernel_size=(3, 3), padding='same', activation='relu')(xs)
    xs = Conv2D(12, kernel_size=(3, 3), padding='same', activation='relu')(xs)

    # M
    xm = Conv2D(20, kernel_size=(7, 7), padding='same', activation='relu')(input1)
    xm = MaxPooling2D(pool_size=(2, 2))(xm)
    xm = Conv2D(40, kernel_size=(5, 5), padding='same', activation='relu')(xm)
    xm = MaxPooling2D(pool_size=(2, 2))(xm)
    xm = Conv2D(20, kernel_size=(5, 5), padding='same', activation='relu')(xm)
    xm = Conv2D(10, kernel_size=(5, 5), padding='same', activation='relu')(xm)

    # L
    xl = Conv2D(16, kernel_size=(9, 9), padding='same', activation='relu')(input1)
    xl = MaxPooling2D(pool_size=(2, 2))(xl)
    xl = Conv2D(32, kernel_size=(7, 7), padding='same', activation='relu')(xl)
    xl = MaxPooling2D(pool_size=(2, 2))(xl)
    xl = Conv2D(16, kernel_size=(7, 7), padding='same', activation='relu')(xl)
    xl = Conv2D(8, kernel_size=(7, 7), padding='same', activation='relu')(xl)

    x = concatenate([xm, xs, xl])
    out = Conv2D(1, kernel_size=(1, 1), padding='same')(x)

    model = Model(inputs=input1, outputs=out)
    model.compile(optimizer=Adam(0.001),
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    return model


def run(model, videopath):
    font = cv2.FONT_HERSHEY_SIMPLEX
    output_text = ''
    writer = None
    sec = 5
    fps = 25.0
    text_color = (0, 255, 0)
    current_dt = datetime.datetime.now()
    cam = cv2.VideoCapture(videopath)
    print('is opened')
    while cam.isOpened():
        ret, frame = cam.read()
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = (gray - 127.5) / 128
        inputs = np.reshape(gray, [1, gray.shape[0], gray.shape[1], 1])
        pred = round(np.sum(model.predict(inputs)))
        curr_time = current_dt.strftime('%Y-%m-%d %H:%M:%S')
        percent_den = int(pred * 100 / MAX_COUNT)
        output_text = str(curr_time) + ' >> PRED : ' + str(pred) + ' '
        print(output_text)
        cv2.rectangle(frame, (10, 1), (800, 45), (0, 0, 0), -1)
        cv2.putText(frame, output_text, (10, 30), font, 0.9, text_color, 3, cv2.LINE_AA)
        frame = cv2.resize(frame, (900, 520))
        writer.write(frame)
    writer.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = get_MCNN()
    model.load_weights('keras_weight/weights_v2.h5')
    videopath = 'icanteen_vid/TEST_2.mp4'
    run(model, videopath)
