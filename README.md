# Hand-Written-Digits-Classifier
A classifier for hand-written digit images using KNN algorithm (no library used) and a side tool to reduce a big dataset to a smaller representative dataset 
using Kmean Clustering (no library again) - First year project in 2018 

#KNN classifier
To start the classifier launch the KNN.java file.
In the KNN.java file you can setup the algorithm as follow:
  -Line 8 -> TESTS is the number of test images you want to classify
  -Line 9 -> K is the hyper parameter the KNN algorithm (the number of neighboor to take into account). If too small you will lose in precision.
  -Lines 10 to 13 -> Here are the different paths to the datasets you want to use.
  -Line 410 -> You can choose which distance metric you want to use either inverted similarity or Euclidean distance. Just swap line 410 with the commented line 413.
 
Once the program is done it will open a window that shows all the test images with their prediction highlighted in green if correct or in red otherwise. 
It also prints you the accuracy and the execution time.
 
#KMeans clustering tool
To use the tool launch the KMeansClustering.java file.
You can setup the program as follow:
  -Line 11 -> K is the size you want for the reduced dataset
  -Line 12 -> maxIters is the maximum number of rounds done by the algorithm.
  -Lines 15-16 -> the paths of the dataset you want to reduce
  -Lines 26-27 -> the paths where to write the new dataset.
  
  
  
 Project done in collaboration with Remi Delacourt for the "Introduction to programming" course of Jamila Sam in 2018.
 
 ENJOY!
    
