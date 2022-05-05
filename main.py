import numpy as np
import cv2
import pandas as pd
from keras.preprocessing import image as img
from keras.layers import Dropout, Flatten, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.models import Model, Sequential


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


def get_target_vector(path, model):
    image = cv2.imread(path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    try:
        (x, y, w, h) = map(lambda x: int(x), face_cascade.detectMultiScale(image, 1.3, 5)[0])
        try:
            margin = (int(h * 0.1), int(w * 0.1))
            face = image[y - margin[0]: y + h + margin[0], x - margin[1]: x + w + margin[1]]
        except:
            face = image[y: y + h, x: x + w]
    except IndexError:
        print('No faces were found')
        raise
    return face, image_to_vector(face, model)


def get_cosine_similarity(a, b):
    return np.dot(a, b) / (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))


model = vgg_face()
model.load_weights('vgg_face_weights.h5')
dataframe = pd.read_pickle('data.pkl')
face, target_vector = get_target_vector('target.jpg', model)
dataframe['similarity'] = dataframe['vector'].apply(lambda vector: get_cosine_similarity(vector, target_vector))
dataframe = dataframe.sort_values(by=['similarity'], ascending=False)
cv2.imshow('face', cv2.resize(face, (500, 500)))
match = dataframe.iloc[0]
cv2.imshow(match['name'] + ' - ' + str(int(match['similarity'] * 100)) + '%', cv2.resize(cv2.imread('imdb_crop/' + match['full_path']), (500, 500)))
cv2.waitKey(0)
