
from sklearn.externals import joblib
import cv2
from para_config import *
from matplotlib import pyplot
import sys
import pandas
import matplotlib.pyplot as plt
import random
import requests
import glob
from shutil import copyfile
import time


clf = joblib.load('02:58PM-coralclassifier.pkl')

# init train_labels
label_data = pandas.read_csv('label_key.csv')
train_labels = []
for index,row in label_data.iterrows():
    train_labels.append(row['label_code'])


if __name__ == '__main__':
# read the image
    time.sleep(10)

    for file_name in glob.glob("*.jpg"):
        # file_name = sys.argv[1]

        time.sleep(5)

        image = cv2.imread(file_name)

        # pos_x = int(row['col'])
        # pos_y = int(row['row'])
        # cur_label = row['label_code']
        #
        # image = image[0:pos_y, 0:pos_x]

        # resize the image
        #image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # predict label of test image
        prediction = clf.predict(global_feature.reshape(1,-1))[0]

        # show predicted label on image
        coral_class = ''
        print 'prediction:' + str(prediction)
        coral_class = label_data[label_data['id'] == prediction]['label_code'].values[0]
        #cv2.putText(image, coral_class, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        temp = str(random.randint(10,14))
        lang = str(random.randint(120,127))
        lat = str(random.randint(68,75))
        depth = str(random.randint(10,30))

        directory = './coral_data/' + coral_class

        if not os.path.exists(directory):
            os.makedirs(directory)

        copyfile(file_name, directory + '/' +file_name)

        url = 'http://coralreefsavior.azurewebsites.net/Home/coralreefdata?temp={0}&lang={1}&lat={2}&depth={3}&coral={4}'.format(temp, lang, lat, depth, coral_class)

        reponse = requests.get(url)

        print 'upload successfully'


        # display the output image
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()