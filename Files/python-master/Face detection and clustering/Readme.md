# Face detection and clustering (2021)
### Using Python with OpenCV function to implement face detection and clustering without existing APIs.

* **Face detection**: Given a face detection dataset composed of hundreds of images, the goal is to detect faces contained in the images. The detector should be able to locate the faces in any testing image.
  1. Method:
     1. Transform image to gray-scale to decrease the noisy.
     2. For face detection, I use Haar Feature-based Cascade Classifier to detect face.
  2. Result: My validation F1 score is 0.8867924528301886. There are somes challenges, such as In some images, it presents a side face instead of a front face, or the face is covered by objects such as a helmet, sunglasses, or other people. This kind of face is hard to been recognize correctly.

* **Face clustering**:  In order to focus on clustering function, the number of clustering is fixed.

  1. Method:
     1. For cropping face, I use above methods to detect face, then saved the boundary box for face_recognition as well as the cropped images for dumping cluster image. 
     2. Use face_recognition to get 128-dimensional face vectors.
     3. Calculate Euclidean distance in every two different feature vectors. 
     4. Use Single Link Cluster to iteratively find the nearest pair and combine together until there are only K clusters.
     
  2. Result:  
     ![image](https://github.com/rosylu/python/blob/master/Face%20detection%20and%20clustering/Result/cluster0.jpg)
