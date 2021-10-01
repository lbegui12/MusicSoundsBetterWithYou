# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:21:00 2021

@author: Louis
"""


# Os  related actions (listing down the file names, checking dir)
import os

#Array Processing
import numpy as np

import keras
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.models import *
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.models import load_model

from midi import read_midi
from midi import convert_to_midi

#importing library
from collections import Counter

#library for visualiation
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split






#specify the path
midi_path='./archive/Alain_Souchon/'        # path to the midi files
best_model_path = "models"                  # path to the already created and trained models
# Check the model dir exists and create it if not
if not os.path.exists(best_model_path):
    os.makedirs(best_model_path)

model_filename = "best_model.h5"
best_model_path = best_model_path + "/" + model_filename 

N_EPOCH = 2             # n_epoch for the model training
create_model = False    # Boolean to determine whether we create the model or load a saved one if possible





# read all the filenames
files=[i for i in os.listdir(midi_path) if i.endswith(".mid")]
print("There are {} files".format(len(files)))

#reading each midi file
notes_array = [] 
for file in files:
    note_array = read_midi(midi_path+file)
    if note_array is not None:
        notes_array.append(note_array)

notes_array = np.array(notes_array) 
print(notes_array.shape)
print(notes_array)


#converting 2D array into 1D array
all_notes = [element for note_ in notes_array for element in note_]



#No. of unique notes
unique_notes = list(set(all_notes))
print("Unique notes = {}".format(len(unique_notes)))




#computing frequency of each note
freq = dict(Counter(all_notes))



#consider only the frequencies
no=[count for _,count in freq.items()]

#set the figure size
plt.figure(figsize=(5,5))

#plot
plt.hist(no)



frequent_notes = [note_ for note_, count in freq.items() if count>=50]
print(len(frequent_notes))


# Update music using only top frequent notes
new_musics=[]

for notes in notes_array:
    temp=[]
    for note_ in notes:
        if note_ in frequent_notes:
            temp.append(note_)            
    new_musics.append(temp)
    
new_musics = np.array(new_musics)


# Preparing the input and output sequences 
no_of_timesteps = 32
x = []
y = []

for song in new_musics:
    print(len(song))
    for i in range(0, len(song) - no_of_timesteps, 1):
        
        #preparing input and output sequences
        input_ = song[i:i + no_of_timesteps]
        output = song[i + no_of_timesteps]
        
        x.append(input_)
        y.append(output)
        
x=np.array(x)
y=np.array(y)

print(x.shape)

# Assigning unique interger to each note
unique_x = list(set(x.ravel()))
x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))



#preparing input sequences (2D)
x_seq=[]
for i in x:
    temp=[]
    for j in i:
        #assigning unique integer to every note
        temp.append(x_note_to_int[j])
    x_seq.append(temp)
    
x_seq = np.array(x_seq)


# same for output (1D)
unique_y = list(set(y))
y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y)) 
y_seq=np.array([y_note_to_int[i] for i in y])



# Prepare test and train data

x_tr, x_val, y_tr, y_val = train_test_split(x_seq,y_seq,test_size=0.2,random_state=0)

x_tr = np.reshape(x_tr, (len(x_tr), no_of_timesteps, 1))
x_val = np.reshape(x_val, (len(x_val), no_of_timesteps, 1))

from keras.utils import to_categorical
y_tr = to_categorical(y_tr)
y_val = to_categorical(y_val)


"""
MODEL CREATION
"""
# Check there is a saved model
if not create_model and os.path.isfile(best_model_path):
    # If so we load the model
    model = load_model(best_model_path)
else:
    # We define and train it
    
    K.clear_session()
    
    model = keras.Sequential()
    model.add(LSTM(
        256,
        input_shape=(x_tr.shape[1], x_tr.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(len(unique_y)))
    
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.summary()
    
    """
    MODEL TRAINING
    """
    # Define Callback to save the best model during training
    mc=ModelCheckpoint(best_model_path, monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    
    # Let's train the model
    model.fit(np.array(x_tr),np.array(y_tr),batch_size=128,epochs=N_EPOCH, validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])


"""
RANDOM MUSIC GENERATION
"""

ind = np.random.randint(0,len(x_val)-1)

random_music = x_val[ind]

predictions=[]
for i in range(30):

    random_music = np.reshape(random_music, (1, no_of_timesteps, 1))
    
    prob  = model.predict(random_music)[0]
    y_pred= np.argmax(prob,axis=0)
    predictions.append(y_pred)

    random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
    random_music = random_music[1:]
    
print(predictions)



# Convert interger back to notes 
x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
predicted_notes = [x_int_to_note[i] for i in predictions]

output_path = "generated_songs/"
filename = "my_generated_song"
convert_to_midi(predicted_notes, output_path, filename)

