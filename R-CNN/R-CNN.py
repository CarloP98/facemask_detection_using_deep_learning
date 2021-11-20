from tensorflow.keras.models import Sequential
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sklearn
import cv2
import os

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
cv2.setUseOptimized(True)

X_mask, Y_mask, X_noMask, Y_noMask, X_other, Y_other  = [],[],[],[],[],[]

def get_iou(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    return intersection_area / float(bb1_area + bb2_area - intersection_area)
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
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.00002),
        metrics=['accuracy']
    )
    return model
def non_max_suppression_fast(boxes, overlapThresh):
   # if there are no boxes, return an empty list

   # if the bounding boxes integers, convert them to floats --
   # this is important since we'll be doing a bunch of divisions

   # initialize the list of picked indexes
   pick = []

   # grab the coordinates of the bounding boxes
   x1 = np.array(boxes)[:,0]
   y1 = np.array(boxes)[:,1]
   x2 = np.array(boxes)[:,2]
   y2 = np.array(boxes)[:,3]

   # compute the area of the bounding boxes and sort the bounding
   # boxes by the bottom-right y-coordinate of the bounding box
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   idxs = np.argsort(y2)

   # keep looping while some indexes still remain in the indexes
   # list
   while len(idxs) > 0:
      # grab the last index in the indexes list and add the
      # index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      # compute the ratio of overlap
      overlap = (w * h) / area[idxs[:last]]

      # delete all indexes from the index list that have
      idxs = np.delete(idxs, np.concatenate(([last],
         np.where(overlap > overlapThresh)[0])))

   final_pick = pick

   # return only the bounding boxes that were picked using the
   # integer data type
   return final_pick

#CREATE DATA
'''
for filename in  os.listdir('cs229_dataset/annotations'):
    name = filename.split('.')[0]
    tree = ET.parse("cs229_dataset/annotations/" + name + '.xml')
    img = cv2.imread("cs229_dataset/images/" + name + '.png')
    mask, locs = [],[]
    root = tree.getroot()
    for elem in root.findall('object'):
        mask.append(1 if elem.find('name').text == 'with_mask' else 0)
        xmin,ymin,xmax,ymax = elem.find('bndbox/xmin').text,elem.find('bndbox/ymin').text,elem.find('bndbox/xmax').text,elem.find('bndbox/ymax').text
        locs.append({"x1": int(xmin), "x2": int(xmax), "y1": int(ymin), "y2": int(ymax)})
    try:
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        maskCounter,noMaskCounter,falsecounter,flag,fflag,bflag = 0,0,0,0,0,0
        for e, result in enumerate(ssresults):
            if e < 10000 and flag == 0:
                for i,loc in enumerate(locs):
                    x, y, w, h = result
                    iou = get_iou(loc, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                    if maskCounter < 40 or noMaskCounter < 40:
                        if iou > 0.75:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            if(mask[i]==1) and (maskCounter < 30):
                                X_mask.append(resized)
                                Y_mask.append([1, 0, 0])
                                maskCounter += 1
                            elif(mask[i]==0) and (noMaskCounter < 30):
                                X_noMask.append(resized)
                                Y_noMask.append([0, 1, 0])
                                noMaskCounter += 1
                    else: fflag = 1
                    if falsecounter < 40:
                        if iou < 0.3:
                            timage = imout[y:y + h, x:x + w]
                            resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                            X_other.append(resized)
                            Y_other.append([0, 0, 1])
                            falsecounter += 1
                    else: bflag = 1
                    if fflag == 1 and bflag == 1:
                        flag = 1
    except Exception as err:
        print(err)
np.save('X_mask', np.array(X_mask))
np.save('Y_mask', np.array(Y_mask))
np.save('X_noMask', np.array(X_noMask))
np.save('Y_noMask', np.array(Y_noMask))
np.save('X_other', np.array(X_other))
np.save('Y_other', np.array(Y_other))

extraData_mask,extraData_noMask = [],[]
extraLabels_mask,extraLabels_noMask = [],[]
for filename in os.listdir('cs229_dataset/moreData/withMask'):
    try:
        image = cv2.imread("cs229_dataset/moreData/withMask/" + filename)
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        extraData_mask.append(resized)
        extraLabels_mask.append([1, 0, 0])
    except Exception as e:
        continue
for filename in os.listdir('cs229_dataset/moreData/withoutMask'):
    try:
        image = cv2.imread("cs229_dataset/moreData/withoutMask/" + filename)
        resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        extraData_noMask.append(resized)
        extraLabels_noMask.append([0, 1, 0])
    except Exception as e:
        continue
np.save('extraX_mask', np.array(extraData_mask))
np.save('extraX_noMask', np.array(extraData_noMask))
np.save('extraY_mask', np.array(extraLabels_mask))
np.save('extraY_noMask', np.array(extraLabels_noMask))
'''

#GET DATA
'''
X_mask = np.concatenate((np.load('X_mask.npy'),np.load('extraX_mask.npy')),axis=0)
Y_mask = np.concatenate((np.load('Y_mask.npy'),np.load('extraY_mask.npy')),axis=0)
X_noMask = np.concatenate((np.load('X_noMask.npy'),np.load('extraX_noMask.npy')),axis=0)
Y_noMask = np.concatenate((np.load('Y_noMask.npy'),np.load('extraY_noMask.npy')),axis=0)
X_other = np.load('X_other.npy')
Y_other = np.load('Y_other.npy')
X_mask,Y_mask = sklearn.utils.shuffle(X_mask,Y_mask)
X_noMask,Y_noMask = sklearn.utils.shuffle(X_noMask,Y_noMask)
X_other,Y_other = sklearn.utils.shuffle(X_other,Y_other)
X = np.concatenate((X_mask,X_noMask,X_other), axis=0)
Y = np.concatenate((Y_mask,Y_noMask,Y_other), axis=0)
X,Y = sklearn.utils.shuffle(X,Y)
split_horizontally_idx = int(X.shape[0]* 0.85)
X_train = X[:split_horizontally_idx , :]
X_test  = X[split_horizontally_idx: , :]
split_horizontally_idx = int(Y.shape[0]* 0.85)
Y_train = Y[:split_horizontally_idx , :]
Y_test  = Y[split_horizontally_idx: , :]
'''
#TRAIN MODEL
'''
#model = build_model()
model = tf.keras.models.load_model('model777.h5')
checkpoint = tf.keras.callbacks.ModelCheckpoint('model888.h5', save_best_only=True)
history = model.fit(
    X_train, Y_train,
    epochs=5,
    batch_size=32,
    verbose=2,
    callbacks=[checkpoint],
    validation_data=(X_test, Y_test)
)

print(model.evaluate(X_test,Y_test))
'''

#TEST
model = tf.keras.models.load_model('model777.h5')

for filename in  os.listdir('testImages'):
    img = cv2.imread('testImages/' + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imout = img.copy()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    noMasksLocations = []
    maskLocations = []
    for e,result in enumerate(ssresults):
        if e < 2000:
            x,y,w,h = result
            timage = img[y:y+h,x:x+w]
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            a = np.expand_dims(resized, axis=0)
            out = model.predict(a)[0]
            if out[0] > 0.85:
                #cv2.imshow('', resized)
                #cv2.waitKey(1000)
                #print([round(out[0], 2), round(out[1], 2), round(out[2], 2)])
                maskLocations.append([x,y,x+w,y+h])
                #cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            if out[1] > 0.85:
                #cv2.imshow('', resized)
                #cv2.waitKey(1000)
                #print([round(out[0], 2), round(out[1], 2), round(out[2], 2)])
                noMasksLocations.append([x, y, x + w, y + h])
                #cv2.rectangle(imout, (x, y), (x+w, y+h), (255, 0, 0), 1, cv2.LINE_AA)

    for i in maskLocations:
        cv2.rectangle(imout, (round(i[0]), i[1]), (i[2], i[3]), (0, 255, 0), 2, cv2.LINE_AA)
    for i in noMasksLocations:
        cv2.rectangle(imout, (round(i[0]), i[1]),(i[2], i[3]), (255, 0, 0), 2, cv2.LINE_AA)

    #merge bounding boxes
    '''
    bb_mask = non_max_suppression_fast(np.asarray(maskLocations), .7) if len(maskLocations) > 0 else []
    bb_noMask = non_max_suppression_fast(np.asarray(noMasksLocations), .7) if len(noMasksLocations) > 0 else []
    for i in bb_mask:
        cv2.rectangle(imout, (round(maskLocations[i][0]), maskLocations[i][1]), (maskLocations[i][2], maskLocations[i][3]), (0, 255, 0), 2, cv2.LINE_AA)
    for i in bb_noMask:
        cv2.rectangle(imout, (round(noMasksLocations[i][0]), noMasksLocations[i][1]), (noMasksLocations[i][2], noMasksLocations[i][3]), (255, 0, 0), 2, cv2.LINE_AA)
    '''
    plt.imshow(imout)
    plt.savefig(filename)
    plt.show()




