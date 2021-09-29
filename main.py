# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 10:21:00 2021

@author: Louis
"""


#for listing down the file names
import os

#Array Processing
import numpy as np

#from keras import Sequential, LSTM, Dense, Activation
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K


from music21 import *

from midi import read_midi

#importing library
from collections import Counter

#library for visualiation
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#specify the path
path='./archive/Alain_Souchon/'

# read all the filenames
files=[i for i in os.listdir(path) if i.endswith(".mid")]
print("There are {} files".format(len(files)))

#reading each midi file
notes_array = [] 
for file in files:
    note_array = read_midi(path+file)
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
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

K.clear_session()

"""
model = Sequential()
    
#embedding layer
model.add(Embedding(len(unique_x), 100, input_length=32,trainable=True)) 

model.add(Conv1D(64,3, padding='causal',activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
    
model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))

model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
          
#model.add(Conv1D(256,5,activation='relu'))    
model.add(GlobalMaxPool1D())
    
model.add(Dense(256, activation='relu'))
model.add(Dense(len(unique_y), activation='softmax'))
    
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.summary()
"""

model = Sequential()
model.add(LSTM(
    256,
    input_shape=(x_tr.shape[1], x_tr.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
#model.summary()
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(len(unique_y)))
#model.summary()
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


"""
MODEL TRAINING
"""
# Define Callback to save the best model during training
mc=ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)

# Let's train the model
history = model.fit(np.array(x_tr),np.array(y_tr),batch_size=128,epochs=20, validation_data=(np.array(x_val),np.array(y_val)),verbose=1, callbacks=[mc])

# Loading the best model
from keras.models import load_model
model = load_model('best_model.h5')



"""
RANDOM MUSIC GENERATION
"""
import random
ind = np.random.randint(0,len(x_val)-1)

random_music = x_val[ind]

predictions=[]
for i in range(30):

    random_music = random_music.reshape(1,no_of_timesteps)

    prob  = model.predict(random_music)[0]
    y_pred= np.argmax(prob,axis=0)
    predictions.append(y_pred)

    random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
    random_music = random_music[1:]
    
print(predictions)



# Convert interger back to notes 
x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x)) 
predicted_notes = [x_int_to_note[i] for i in predictions]



# Function to convert back to notes
def convert_to_midi(prediction_output):
   
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:
            
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='music.mid')


convert_to_midi(predicted_notes)

