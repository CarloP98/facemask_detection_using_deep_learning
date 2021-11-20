# Positive Examples
import time
start_time = time.time()

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

############################ Load Train and Test Data ##############################################
from skimage import data, transform , color
import matplotlib.pyplot as plt
import cv2
from numpy import genfromtxt
import numpy as np
training_data_path = "C:\\Users\\m_pou\\OneDrive\\Documents\\Data_Model_Optimization Certificate Program\\Project\\CNN_Train_Img_Lbl_postprocessed\\"
testing_data_path = "C:\\Users\\m_pou\\OneDrive\\Documents\\Data_Model_Optimization Certificate Program\\Project\\CNN_Test_Img_Lbl_postprocessed\\"
label_name_train = training_data_path + 'train_label.csv'
label_name_test = testing_data_path + 'test_label.csv'

train_label = genfromtxt(label_name_train, delimiter=',')
test_label = genfromtxt(label_name_test, delimiter=',')

final_train_images = []
for i in range(len(train_label)):
    train_image_path = training_data_path + 'train_image_' + str(i) + '.png'
    final_train_images_4ch = plt.imread(train_image_path)
    final_train_images_3ch = final_train_images_4ch[:, :, 0:3]
    final_train_images_3ch = cv2.resize(final_train_images_3ch, (62,62), interpolation = cv2.INTER_AREA)
    final_train_images_3ch = color.rgb2gray(final_train_images_3ch)
    final_train_images.append(final_train_images_3ch)

final_test_images = []
for i in range(len(test_label)):
    test_image_path = testing_data_path + 'test_image_' + str(i) + '.png'
    final_test_images_4ch = plt.imread(test_image_path)
    final_test_images_3ch = final_test_images_4ch[:, :, 0:3]
    final_test_images_3ch = cv2.resize(final_test_images_3ch, (62,62), interpolation = cv2.INTER_AREA)
    final_test_images_3ch = color.rgb2gray(final_test_images_3ch)
    final_test_images.append(final_test_images_3ch)

train_image = np.array(final_train_images)
test_image = np.array(final_test_images)
###############################################

# Combine sets and extract HOG features
from skimage.feature import hog
from itertools import chain

X_train  = np.array([hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualize=False) for im in train_image])
X_test  = np.array([hog(im, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualize=False) for im in test_image])

y_train = train_label
y_test = test_label
print(X_train.shape)
print(y_train.shape)

# Training a support vector machine
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
model = LinearSVC()
model.fit(X_train, y_train)
print("######################   Train Performance  ###########################")
print(cross_val_score(LinearSVC(), X_train, y_train))

print("######################   Test Performance ###########################")
print(cross_val_score(LinearSVC(), X_test, y_test))

##################   Sliding Window and K-means ###################################
def sliding_window(img, patch_size=train_image[0].shape,
                   istep=1, jstep=1, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch

# print("--- %s seconds ---" % (time.time() - start_time))

# Selecting a test figure and rescaling it
def face_localize(model, test_image , image_scale_factor , patch_size=train_image[0].shape , max_face_num = 3):
    import skimage
    #test_image = skimage.data.astronaut()

    test_image = skimage.color.rgb2gray(test_image)
    test_image = skimage.transform.rescale(test_image, image_scale_factor)

    # Sliding Window and detecting the windows with positive prediction
    indices, patches = zip(*sliding_window(test_image))
    patches_hog = np.array([hog(patch, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualize=False) for patch in patches])
    patches_hog.shape

    labels = model.predict(patches_hog)
    confidence = model._predict_proba_lr(patches_hog)

    labels.sum()

    # Drawing Bounding box
    Ni, Nj = patch_size
    indices = np.array(indices)



    # Draw all the bounding boxes with label = 1
    fig, ax = plt.subplots()
    ax.imshow(test_image)
    ax.axis('off')
    for i, j in indices[labels == 1]:
        ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                                   alpha=0.3, lw=2, facecolor='none'))

    """
    # Draw all the bounding boxes center points with label = 1
    fig, ax = plt.subplots()
    ax.imshow(test_image)
    ax.axis('off')
    for i, j in indices[labels == 1]:
        ax.add_patch(plt.Rectangle((j+Nj/2, i+Ni/2), 1, 1, edgecolor='red',
                                   alpha=0.3, lw=4, facecolor='none'))
    """

     # k-mean clustring algorithm for multi-face detection
    import random as rand
    ##bb_index = indices[labels == 1]
    bb_index = indices[confidence[:, 1] > 0.60]
    bb_index_list = list(bb_index)
    mu  = []
    iter = 0

    for j in range(max_face_num):
        mu.append(rand.choice(bb_index_list))

    mu_p = mu
    while iter<200:
        iter = iter + 1
        c = np.zeros(len(bb_index_list))
        for i in range(len(bb_index_list)) :
            dist = np.zeros(max_face_num)
            for j in range(max_face_num):
                dist[j] = np.linalg.norm(mu[j]-bb_index_list[i])**2
            c[i] = np.argmin(dist)

        mu  = []
        for j in range(max_face_num):
            mu_j_cum = 0
            mu_j_count = 0
            for i in range(len(bb_index_list)):
                mu_j_cum = mu_j_cum + (c[i] == j) * bb_index_list[i]
                mu_j_count = mu_j_count + (c[i] == j) * 1
            mu_j = np.floor(mu_j_cum / mu_j_count)
            mu.append(mu_j)
            mu_error = 0
        for error_count in range(len(mu)):
            mu_error = mu_error + np.linalg.norm(mu[error_count]-mu_p[error_count])
        mu_p = mu

    selected_mu = []
    actual_face_number = 0
    for j in range(0,max_face_num-1):
        select_flag = 1
        for jj in range(j+1, max_face_num):
            if np.abs(mu[j][0]- mu[int(jj)][0])<Ni and np.abs(mu[j][1]- mu[int(jj)][1])<Nj:
                select_flag = 0
        if select_flag == 1:
            selected_mu.append(mu[j])

    select_flag = 1
    for jj in range(0, len(selected_mu)):
        if np.abs(mu[max_face_num-1][0] - selected_mu[int(jj)][0]) < Ni and np.abs(mu[max_face_num-1][1] - selected_mu[int(jj)][1]) < Nj:
            select_flag = 0
    if select_flag == 1:
        selected_mu.append(mu[max_face_num-1])

    return selected_mu, Ni, Nj , confidence , labels

##################################################################
unknow_test_image= plt.imread('img_352.jpg')

test_image_size = unknow_test_image.shape
patch_size=train_image[0].shape
image_scale_factor = (6**0.5)*max(patch_size[0]/test_image_size[0] , patch_size[1]/test_image_size[1])
selected_mu, Ni, Nj , confidence , labels= face_localize(model, unknow_test_image, image_scale_factor,
                                                           patch_size=train_image[0].shape, max_face_num=3)
fig, ax = plt.subplots()
ax.imshow(unknow_test_image)
ax.axis('off')
for jj in range(len(selected_mu)):
    ax.add_patch(plt.Rectangle([selected_mu[jj][1]/image_scale_factor, selected_mu[jj][0]/image_scale_factor], Nj/image_scale_factor, Ni/image_scale_factor, edgecolor='red', alpha=0.3, lw=4,
                               facecolor='none'))