import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

#---------------Libraries specifically for Task 2---------------#
from tqdm.notebook import tqdm
from skimage import feature       
from sklearn.svm import LinearSVC                   #Clf1
from sklearn.ensemble import RandomForestClassifier #Clf2
import joblib
import pandas as pd               
import plotly.graph_objects as go                   #For Data Visualization

#function to load images and return the 'path + image' name
#filetype = '.png' , '.jpg' or '.jpeg'
def loadImage(path,filetype):
  images = [os.path.join(path,file)
  for file in os.listdir(path)
   if file.endswith(filetype)]
  return images

#---------------FUNCTIONS for Task 1---------------#

#function to display comparison image with customized window size (ixj)
def compareImage(image,imgtitle,i=8,j=6):
  """
  # This function takes images along with its title
  # and returns a figure of customized dimensions (ixj)
  # Args:
  #   image: 1st image (numpy array)
  #   title: title of image (string) 
  #   i,j: dimension of figure (int) {default ixj = 8x6}
  # Returns:
  #   an image showing feature matching in both input images

  # """
  plt.figure(figsize=(i,j))
  plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB))
  plt.xticks([]), plt.yticks([]) #Axis Markers turned off
  plt.title(imgtitle)
  plt.show()


#function to Compute "Oriented FAST and Rotated BRIEF (ORB)" and Match Features
def computeORBnMatch(img1,img2,i):
  """
  # This function takes two images, detects keypoints and computes descriptors using ORB
  # and returns the output image showing desired number of features being matched.
  # Args:
  #   img1: a grayscale image (numpy array)
  #   img2: a grayscale image (numpy array) 
  #   i: no. of features to be matched (int)
  # Returns:
  #   an image showing feature matching in both input images

  """
  # Initiate ORB detector
  orb = cv.ORB_create()
  
  # Find the keypoints and descriptors with SIFT
  kp1, des1 = orb.detectAndCompute(img1,None)
  kp2, des2 = orb.detectAndCompute(img2,None)
  
  # Create BFMatcher object
  bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
  
  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 'i' no. of matches.
  out = cv.drawMatches(img1,kp1,img2,kp2,matches[:i], None,flags=2)
  print('ORB applied and Features Matched')
  return out


#function to Compute "Scale Invariant Feature Transform (SIFT)" and Match Features
def computeSIFTnMatch(img1,img2,i):
  """
  # This function takes two images, detects keypoints and computes descriptors using SIFT
  # and returns the output image showing desired number of features being matched.
  # Args:
  #   img1: a grayscale image (numpy array)
  #   img2: a grayscale image (numpy array) 
  #   i: no. of features to be matched (int)
  # Returns:
  #   an image showing feature matching in both input images
  """
  # Initiate SIFT detector
  sift = cv.xfeatures2d.SIFT_create()

  # Find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  # Create BFMatcher object
  bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 'i' no. of matches.
  out = cv.drawMatches(img1,kp1,img2,kp2,matches[:i], None,flags=2)
  print('SIFT applied and Features Matched')
  return out


#function to Compute "Speeded-Up Robust Features (SURF) and Match Features"
def computeSURFnMatch(img1,img2,i):
  """
  # This function takes two images, detects keypoints and computes descriptors using ORB
  # and returns the output image showing desired number of features being matched.
  # Args:
  #   img1: a grayscale image (numpy array)
  #   img2: a grayscale image (numpy array) 
  #   i: no. of features to be matched (int)
  # Returns:
  #   an image showing feature matching in both input images
  
  """
  # Initiate SURF detector
  surf = cv.xfeatures2d.SURF_create()

  # Find the keypoints and descriptors with SIFT
  kp1, des1 = surf.detectAndCompute(img1,None)
  kp2, des2 = surf.detectAndCompute(img2,None)

  # Create BFMatcher object
  bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 'i' no. of matches.
  out = cv.drawMatches(img1,kp1,img2,kp2,matches[:i], None,flags=2)
  print('SURF applied and Features Matched')
  return out

#---------------FUNCTIONS for Task 2---------------#

#function to read and process images, and store them and their respective labels in 2 different lists
def DataPreprocessing(imgArr,imgArr2,x,y,lbl,lbl2):
  """
  # This function takes path+file name of images (from dataset of 2 classes respectively)
  # reads them in grayscale, and stores them after resizing. As an output, two arrays are returned,
  # one having the processed images and the other containing the respective labels.
  # Args:
  #   imgArr: a list containing Path+Filenames of Dataset Images of 1st Class (numpy array)
  #   imgArr2: a list containing Path+Filenames of Dataset Images of 2nd Class (numpy array)
  #   x,y: x by y dimensions for resizing image (int) 
  #   lbl: label to be assigned to 1st Class (int)
  #   lbl2: label to be assigned to 2nd Class (int)
  # Returns:
  #   newImgArr: a list containing processed DataSet (images) of both classes  
  #   lblArr: a list containing lebels for dataset (images) of both classes

  """
  newImgArr = []
  lblArr = []
  for i in tqdm(range(0,len(imgArr))):
    img = cv.imread(imgArr[i],0)
    img = cv.resize(img, (x, y))
    newImgArr.append(img)
    lblArr.append(lbl)

  for i in tqdm(range(0,len(imgArr2))):
    img = cv.imread(imgArr2[i],0)
    img = cv.resize(img, (x, y))
    newImgArr.append(img)
    lblArr.append(lbl2)
  
  return newImgArr, lblArr

#Compute HoG Features
def compute_HOG(image):
  (H, hogImage) = feature.hog(image, orientations = 9,
                                pixels_per_cell  = (8, 8), cells_per_block  = (2, 2), transform_sqrt=True,
                                block_norm  = 'L1' , visualize=True)
  return (H, hogImage)

#function to Construct Confusion Matrix"
def constructConfusionMatrix(actual_labels,predicted_labels):
  """
  # This function takes actual labels and predicted labels
  # and constructs a confusion matrix
  # Args:
  #   predicted_labels: predicted labels (numpy array)
  #   actual_labels: actual labels (numpy array) 

  # Returns:
  #   Data Frame depicting a Confusion Matrix (3x3 Matrix for 2 classes)

  """
  act = pd.Series(actual_labels,name='Actual')
  pred = pd.Series(predicted_labels,name='Predicted')
  confusion_matrix = pd.crosstab(act, pred, margins=True)
  print("Confusion matrix:\n%s" % confusion_matrix)
  return confusion_matrix


#function to Plot Confusion Matrix"
def plotConfusionMatrix(df_confusion, c_map = 'YlOrRd'):
  """
  # This function takes Confusion Matrix data frame as input
  # and plots a confusion matrix plot
  # Args:
  #   df_confusion: Confusion Matrix Data frame
  #   cmap: Cmap value (string) {Default = YlOrRd}
  # Returns:
  #   None
  
  """
  plt.matshow(df_confusion, cmap = c_map) # imshow
  plt.colorbar()
  tick_marks = np.arange(len(df_confusion.columns))
  plt.xticks(tick_marks, df_confusion.columns, rotation=45)
  plt.yticks(tick_marks, df_confusion.index)
  plt.ylabel(df_confusion.index.name)
  plt.xlabel(df_confusion.columns.name)


#function to Compute Performance Measures of a Classifier"
def computePerformanceMeasures(CM,x=0):
  """
  # This function takes Confusion Matrix data frame as input
  # and outputs various performance indicators/measures
  # Args:
  #   CM : confusion matrix (Data Frame)
  #   x : check to display TP|FP|TN|FN values (int) {Default x=0, so no output}
  # Returns:
  #   TPR : True Positive Rate (float)
  #   FPR : False Positive Rate (float)
  #   FS : F1 Score (float)
  #   AC : Accuracy (float)
  #   Only the required parameters have been returned as output
  """
  TN = CM[0][0] # True  Negative
  FN = CM[1][0] # False Negative
  TP = CM[1][1] # True  Positive
  FP = CM[0][1] # False Positive

  if x == 1:
    print('--------------------------------------------')
    print("True Negative: ", TN, " | False Negative: ", FN)
    print("True Positive: ", TP, "| False Positive: ", FP)
    print('--------------------------------------------')

  # True Positive Rate | Sensitivity | Hit rate | Recall
  TPR = TP/(TP+FN)
  # False Positive Rate | Fall out
  FPR = FP/(FP+TN)

  # True Negative Rate | Specificity
  TNR = TN/(TN+FP) 
  # False Negative Rate
  FNR = FN/(TP+FN)

  # Positive Predictive Value |  Precision
  PPV = TP/(TP+FP)
  # Negative Predictive Value
  NPV = TN/(TN+FN)
  # False Discovery Rate
  FDR = FP/(TP+FP)

  # F1 Score
  FS = 2*(PPV*TPR)/(PPV+TPR)
  # Overall Accuracy
  AC = (TP+TN)/(TP+FP+FN+TN)

  # print("True Positive Rate: ", round(TPR,2), "| F1 Score: ", round(FS,2))
  # print("False Positive Rate: ", round(FPR,2), "| Accuracy: ", round(AC,2))

  return TPR, FPR, FS, AC


#function to compare Performance Measures of 2 Classifiers in Tabular Form
def compareTwoClassifiers(C1,TPR1,FPR1,FS1,AC1,C2,TPR2,FPR2,FS2,AC2):
  """
  # This function takes various parameters for 2 different classifiers
  # and plot a tabular form for comparison
  # Args:
  # 1st Classifier Parameters
  #   C1 : confusion matrix (Data Frame)
  #   TPR1 : True Positive Rate (float)
  #   FPR1 : False Positive Rate (float)
  #   FS1 : F1 Score (float)
  #   AC1 : Accuracy (float)
  # 2nd Classifier Parameters
  #   C2 : confusion matrix  (Data Frame)
  #   TPR2 : True Positive Rate (float)
  #   FPR2 : False Positive Rate (float)
  #   FS2 : F1 Score (float)
  #   AC2 : Accuracy (float)
  # Returns:
  #   A figure showing comparison of above parameters in tabular form
  """
  fig = go.Figure(data=[go.Table(
                  header=dict(
                    values=['Performance Measures',C1,C2],
                    line_color='darkslategray',
                    fill_color='royalblue',
                    font=dict(color='white', size=14)
                             ),

                  cells=dict(
                    values=[['True Positive Rate', 'False Positive Rate', 'F1 Score', 'Accuracy'], 
                            [round(TPR1,2), round(FPR1,2), round(FS1,2), round(AC1,2)],
                            [round(TPR2,2), round(FPR2,2), round(FS2,2), round(AC2,2)]],
                    line_color='darkslategray',
                    fill=dict(color=['paleturquoise', 'white']))
                                )
                       ]
                 )
  fig.show()


  #function to display 2 images in 1x2 Subplot
def displayTwoImages(image1,image2,title1,title2):
  """
  # This function takes two images along with their titles
  # and displays a subplot containing both images (in grayscale)
  # Args:
  #   img1: 1st image (numpy array)
  #   img2: 2nd image (numpy array) 
  #   title1: title of image1 (string)
  #   title2: title of image2 (string) 
  # Returns:
  #   None

  """
  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.imshow(image1, cmap = 'gray')
  plt.title(title1), plt.xticks([]), plt.yticks([]) #Axis Markers turned off

  fig.add_subplot(1, 2, 2)
  plt.imshow(image2, cmap = 'gray')
  plt.title(title2), plt.xticks([]), plt.yticks([]) #Axis Markers turned off
  plt.show()
