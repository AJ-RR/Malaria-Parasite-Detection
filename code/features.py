import cv2
import os
import numpy as np
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

""" ######################## Helper Functions (BLOB) ######################## """
def adjust_gamma(image, gamma=1.0):
    	# build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def load_dataset(path, classes) :
    class_images = []
    for cls in classes :
        cls_imgs = []
        img_names = os.listdir(path + cls + "/")
        for img_name in img_names :
                try :
                    img = cv2.imread(path + cls + "/" + img_name)
                    if not (img is None):
                        cls_imgs.append(img)
                except Exception as e :
                    pass
        class_images.append(np.array(cls_imgs))
    return np.concatenate(class_images)

""" ######################## ############### ########################## """
""" ######################## Hyperparameters (BLOB) ######################## """
# Blob detection (Hyperparameters)
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.maxArea = 200
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.3
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.0
# Change thresholds
params.minThreshold = -3;
params.maxThreshold = 150;
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.0
detector = cv2.SimpleBlobDetector_create(params)
# other hyperparameters
IMG_SIZE = 90
gamma = 0.95
smoothening_kernel_size = 5
smoothening_degree = 50
""" ######################## ############### ########################## """



# takes in the path of the image files and returns a dataframe with the 5 features
def get_contour_features(dataset):
    dataframe = [[],[],[],[],[]]
    for img in dataset:
        img_ = cv2.GaussianBlur(img, (3,3), 2)
        if not (img_ is None):
            img_gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(img_gray,127,255,0)
            _,contours,_ = cv2.findContours(thresh,1,2)

            for i in range(5):
                try:
                    area = cv2.contourArea(contours[i])
                    dataframe[i].append(area)
                except:
                    dataframe[i].append(0)

    return (np.asarray(dataframe).T)

def get_number_of_contours(dataset):
    dataframe = []
    for img in dataset:
        if not (img is None):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30 , 200)
            _,contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            dataframe.append(len(contours))
    return (np.asarray(dataframe).T)

# takes in the path of the image files and returns a dataframe with the number of blobs
def get_blob_features(dataset):
    dataframe = [[],[],[]]
    for img in dataset:
        if not (img is None):
            img_ = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            img_ = cv2.dilate(img_,kernel,iterations = 1)
            img_ = adjust_gamma(img_, gamma)
            keypoints = detector.detect(img_)
            dataframe[0].append(len(keypoints))
            if(len(keypoints) >= 2):
                dataframe[1].append(keypoints[0].size)
                dataframe[2].append(keypoints[1].size)
            elif(len(keypoints) == 1):
                dataframe[1].append(keypoints[0].size)
                dataframe[2].append(0)
            else:
                dataframe[1].append(0)
                dataframe[2].append(0)
    return np.asarray(dataframe).T

def generate_LUCID_features(train_img):
    orb_keypoints = []
    count = 0
    rmv_index_train = []
    for image in train_img :
         if not (image is None):
            image = cv2.resize(image, (50, 50))
            # image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            detector = cv2.FastFeatureDetector_create()
            kp = detector.detect(image,None)
            orb = cv2.xfeatures2d.LUCID_create()
            kp, descriptors = orb.compute(image,kp)
            if descriptors is not None:
                descriptors = np.array(descriptors)
                orb_keypoints.append(descriptors)

                temp = descriptors
            else:

                orb_keypoints.append(temp)
                # rmv_index_train.append(count)

            # count = count + 1
    orb_keypoints = np.concatenate(orb_keypoints, axis=0)
    kmeans = KMeans(n_clusters = 16).fit(orb_keypoints)
    print(orb_keypoints.shape)
    print("--------Computed descriptors--------")

    x_Siftfeat_train = calculate_lucid_histogram(train_img, kmeans)
    print("------Computed Histogram-----")

    return x_Siftfeat_train
def calculate_lucid_histogram(images, model):
    feature_vectors=[]
    rmv_index_test = []
    for image in images :
        if not (image is None):
            # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #SIFT extraction
            detector = cv2.FastFeatureDetector_create()
            kp = detector.detect(image,None)
            orb = cv2.xfeatures2d.LUCID_create()
            kp, descriptors = orb.compute(image,kp)
            #classification of all descriptors in the model
            if descriptors is not None :
                predict_kmeans = model.predict(descriptors)
                #calculates the histogram
                hist, bin_edges = np.histogram(predict_kmeans, bins = 16)
                #histogram is the feature vector
                feature_vectors.append(hist)
                temp = hist
            else :
                feature_vectors.append(temp)
                # rmv_index_test.append(count)

    feature_vectors=np.asarray(feature_vectors)

    return np.array(feature_vectors)


def get_features():
    DATADIR = "/home/redhood/Desktop/CollegeWork/Semester_5/Machine_Learning/Rough/Group_Project/Parasite/Parasite/train/"
    dataset = load_dataset(DATADIR, ["Uninfected", "Parasitized"])
    
    print("Type 1 to get contour features: ")
    x = int(input())
    print("Type 1 to get blob features: ")
    y = int(input())
    print("Type 1 to get LUCID features: ")
    z = int(input())
    print("Type 1 to get number of contours: ")
    y_ = int(input())

    if(x == 1):
        data1 = get_contour_features(dataset)
        df1 = pd.DataFrame(data = data1)
        df1.to_csv("contour_features.csv")
    #print(data1.shape)
    
    if(y == 1):
        data2 = get_blob_features(dataset)
        df2 = pd.DataFrame(data = data2)
        df2.to_csv("blob_features.csv")
    
    #print(data2.shape)
    if(z == 1):
        data3 = generate_LUCID_features(dataset)
        df3 = pd.DataFrame(data = data3)
        df3.to_csv("LUCID_features.csv")
    
    if(y_ == 1):
    #print(data3.shape)
        data4 = get_number_of_contours(dataset)
        df4 = pd.DataFrame(data = data4)
        df4.to_csv("num_of_contours.csv")



def main():
    get_features()


# sync; echo 3 > /proc/sys/vm/drop_caches - to clear RAM

if __name__ == "__main__" :
    main()
