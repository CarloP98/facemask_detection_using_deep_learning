from tensorflow.keras.models import Sequential
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import cv2
import os

def build_model():
    mobilenet = tf.keras.applications.MobileNetV2(
        weights='mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    model = Sequential()
    model.add(mobilenet)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.00002),
        metrics=['accuracy']
    )
    return model

#GET DATA
'''
X = np.concatenate((np.load('X_mask.npy'), np.load('X_noMask.npy')), axis=0)
Y = np.concatenate((np.load('Y_mask.npy'), np.load('Y_noMask.npy')), axis=0)

index = 0
idx = []
for i in Y:
    if i[2] == 1: idx.append(index)
    index += 1

X = np.delete(X, idx, axis=0)
Y = np.delete(Y, idx, axis=0)
Y = np.delete(Y, -1, axis=1)
X,Y = sklearn.utils.shuffle(X, Y)

split_horizontally_idx = int(X.shape[0]* 0.8)
X_train = X[:split_horizontally_idx , :]
X_test  = X[split_horizontally_idx: , :]
split_horizontally_idx = int(Y.shape[0]* 0.8)
Y_train = Y[:split_horizontally_idx , :]
Y_test  = Y[split_horizontally_idx: , :]
'''

#TRAIN MODEL
'''
model = build_model()
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)
history = model.fit(
    X_train, Y_train,
    epochs=15,
    batch_size=32,
    verbose=2,
    callbacks=[checkpoint],
    validation_data=(X_test, Y_test)
)

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

print(model.evaluate(X_test,Y_test))
'''

#TEST
model = tf.keras.models.load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for filename in  os.listdir('testImages'):
    img = cv2.imread('testImages/' + filename)
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:
        face = img[y:y + w, x:x + w]
        resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        exp = np.expand_dims(resized, axis=0)
        out = model.predict(exp)
        print(out)
        max = np.argmax(out[0])
        if max == 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, "mask " + str(round(out[0][0]*100,1)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, min(w,h)/200, (0,255,0), 1)
        elif max == 1:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "no_mask "  + str(round(out[0][1]*100,1)) + "%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, min(w,h)/200, (0, 0, 255), 1)
    cv2.imwrite(filename,img)
