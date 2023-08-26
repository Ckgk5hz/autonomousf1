# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('final_training_data.npy', allow_pickle = True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))


lefts = []
rights = []
forwards = []
backwards = []
forwardleft = []
forwardright = []
backwardleft = []
backwardright =[]

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0,0,0,0,0,0]:
        lefts.append([img,choice])

    elif choice == [0,1,0,0,0,0,0,0]:
        forwards.append([img,choice])

    elif choice == [0,0,1,0,0,0,0,0]:
        backwards.append([img,choice])

    elif choice == [0,0,0,1,0,0,0,0]:
        rights.append([img,choice])

    elif choice == [0,0,0,0,1,0,0,0]:
        forwardleft.append([img,choice])

    elif choice == [0,0,0,0,0,1,0,0]:
        forwardright.append([img,choice])

    elif choice == [0,0,0,0,0,0,1,0]:
        backwardleft.append([img,choice])

    elif choice == [0,0,0,0,0,0,0,1]:
        backwardright.append([img,choice])

    else:
        print('no matches')


forwards = forwards[:len(lefts)][:len(rights)][:len(backwards)][:len(forwardleft)][:len(forwardright)][:len(backwardleft)][:len(backwardright)]
lefts = lefts[:len(forwards)]
rights = rights[:len(forwards)]
backwards = backwards[:len(forwards)]
forwardleft = forwardleft[:len(forwards)]
forwardright = forwardright[:len(forwards)]
backwardleft = backwardleft[:len(forwards)]
backwardright = backwardright[:len(forwards)]

final_data = forwards + lefts + rights + backwards + forwardright + forwardleft + backwardleft + backwardright

shuffle(final_data)

print(len(final_data))

np.save('balanced_final_data_again.npy', final_data)