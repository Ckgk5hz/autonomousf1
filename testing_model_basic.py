# testing_model_basic.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
from alexnet import alexnet
import os
from directkeys import PressKey, ReleaseKey, W, A, S, D

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 7
MODEL_NAME = 'f1-{}-{}-{}-withthingies'.format(LR, 'alexnetv2',EPOCHS)

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def left():
    ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def right():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(D)

def brakes():
    ReleaseKey(W)
    ReleaseKey(A)
    PressKey(S)
    ReleaseKey(D)

def fleft():
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def fright():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(D)

def bleft():
    ReleaseKey(W)
    PressKey(A)
    PressKey(S)
    ReleaseKey(D)

def bright():
    ReleaseKey(W)
    ReleaseKey(A)
    PressKey(S)
    PressKey(D)
 

model=alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    paused = False

    while(True):

        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            # resize to something a bit more acceptable for a CNN
            
            last_time = time.time()
            prediction = model.predict([screen.reshape(WIDTH,HEIGHT,1)])[0]
            moves = list(np.around(prediction))
            print(moves,prediction)

            if moves == [1,0,0,0,0,0,0,0]:
                left()
            elif moves == [0,1,0,0,0,0,0,0]:
                brakes()
            elif moves == [0,0,1,0,0,0,0,0]:
                straight()
            elif moves == [0,0,0,1,0,0,0,0]:
                right()
            elif moves == [0,0,0,0,1,0,0,0]:
                fleft()
            elif moves == [0,0,0,0,0,1,0,0]:
                fright()
            elif moves == [0,0,0,0,0,0,1,0]:
                bleft()
            elif moves == [0,0,0,0,0,0,0,1]:
                bright()
            
        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused=True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(S)
                ReleaseKey(D)
                time.sleep(1)


main()