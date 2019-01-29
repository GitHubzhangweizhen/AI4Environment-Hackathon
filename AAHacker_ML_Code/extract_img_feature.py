#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import pandas
from para_config import *



# # get the training labels
# train_labels = os.listdir(train_path)
#
# # sort the training labels
# train_labels.sort()
# print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 80

#load training data
train_data = pandas.read_csv('training_dataset.csv')
label_data = pandas.read_csv('label_key.csv')
label_data = label_data.drop(['functional_group','label_description','benchmark'], axis=1)
train_data = train_data.join(label_data.set_index('label_code'), on='label_code')
train_data = train_data[(train_data['functional_group'] != 'Other') & (train_data['functional_group'] != 'Other_Inv')]
#train_data = train_data.sample(n=100)

for index,row in train_data.iterrows():
    file_name = row['file_name']
    pos_x = int(row['col'])
    pos_y = int(row['row'])
    cur_label = row['id']

    image = cv2.imread(file_name)
    image = image[0:pos_y, 0:pos_x]
    #cv2.imshow('crop_image',image)

    #TODO image should be resized

    # Image Feature extraction
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    labels.append(cur_label)
    global_features.append(global_feature)

    print "[STATUS] processed qid: {}".format(row['qid'])


# loop over the training data sub-folders
# for training_name in train_labels:
#     # join the training data path and each species training folder
#     dir = os.path.join(train_path, training_name)
#
#     # get the current training label
#     current_label = training_name
#
#     k = 1
#     # loop over the images in each sub-folder
#     for x in range(1,images_per_class+1):
#         # get the image file name
#         file = dir + "/" + str(x) + ".jpg"
#
#         # read the image and resize it to a fixed-size
#         image = cv2.imread(file)
#         image = cv2.resize(image, fixed_size)
#
#         ####################################
#         # Global Feature extraction
#         ####################################
#         fv_hu_moments = fd_hu_moments(image)
#         fv_haralick   = fd_haralick(image)
#         fv_histogram  = fd_histogram(image)
#
#         ###################################
#         # Concatenate global features
#         ###################################
#         global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
#
#         # update the list of labels and feature vectors
#         labels.append(current_label)
#         global_features.append(global_feature)
#
#         i += 1
#         k += 1
#     print "[STATUS] processed folder: {}".format(current_label)
#     j += 1

print "[STATUS] completed Global Feature Extraction..."

# get the overall feature vector size
print "[STATUS] feature vector size {}".format(np.array(global_features).shape)

# get the overall training label size
print "[STATUS] training Labels {}".format(np.array(labels).shape)

# encode the target labels
# targetNames = np.unique(labels)
# le = LabelEncoder()
# target = le.fit_transform(labels)
target = np.array(labels)
print "[STATUS] training labels encoded..."

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print "[STATUS] feature vector normalized..."

print "[STATUS] target labels: {}".format(target)
print "[STATUS] target labels shape: {}".format(target.shape)

# save the feature vector using HDF5
h5f_data = h5py.File('output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print "[STATUS] end of training.."