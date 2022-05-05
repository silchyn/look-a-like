import scipy.io
import pandas as pd
import numpy as np
import cv2
from keras.preprocessing import image as img
from keras.layers import Dropout, Flatten, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import Model, Sequential


def process_raw_data():
    meta = scipy.io.loadmat('imdb_crop/imdb.mat')
    all_columns = ['dob', 'photo_taken', 'full_path', 'gender', 'name', 'face_location',
                   'face_score', 'second_face_score', 'celeb_names', 'celeb_id']
    columns = ['full_path', 'name', 'face_score', 'second_face_score']
    df = pd.DataFrame(index=range(meta['imdb'][0][0][0].shape[1]), columns=columns)
    for (i, column) in enumerate(meta['imdb'][0][0]):
        if all_columns[i] in columns:
            df[all_columns[i]] = pd.DataFrame(column[0])
    df = df[((df['face_score'] != -np.inf) & df['second_face_score'].isna()) & (df['face_score'] >= 3)]
    df['full_path'] = df['full_path'].apply(lambda path: path[0])
    df['name'] = df['name'].apply(lambda name: name[0])
    return df[['full_path', 'name']]


def vgg_face():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    return vgg_face_descriptor


def image_to_vector(image, model):
    image = img.img_to_array(cv2.resize(image, (224, 224)))
    image = np.expand_dims(image, axis=0) / 127.5 - 1
    return model.predict(image)[0, :]


model = vgg_face()
model.load_weights('vgg_face_weights.h5')
data = process_raw_data()
data['vector'] = data['full_path'].apply(lambda path: image_to_vector(cv2.imread('imdb_crop/' + path), model))
data.to_pickle('data.pkl')
