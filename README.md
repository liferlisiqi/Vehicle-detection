## Vehicle Detection Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/cutout1_HOG.jpg
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/label.png
[image7]: ./output_images/bboxes.png
[image8]: ./output_images/1.png
[image9]: ./output_images/5.png
[image10]: ./output_images/6.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### HOG and color features
#### 1. Extracte HOG features from the training images.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then I use skimage.feature.hog() to extract HOG features.
```python3
hog(img, orientations=orient,pixels_per_cell=(pix_per_cell,pix_per_cell),cells_per_block=(cell_per_block,cell_per_block),transform_sqrt=False, visualise=vis, feature_vector=feature_vec) 
```

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image2]

I tried various combinations of parameters and `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` was selected as the final choose.

#### 2. Extracte color features from the training images.
Extract histogram and binned color features of each channel.
```python
def color_hist(img, nbins=32):
    hist1 = np.histogram(img[:,:,0], bins=nbins)
    hist2 = np.histogram(img[:,:,1], bins=nbins)
    hist3 = np.histogram(img[:,:,2], bins=nbins)
    
    # concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
    
    return hist_features
```
```python
def bin_spatial(img, size=(32,32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    
    return np.hstack((color1, color2, color3))
```

#### 3. Traine a classifier using the selected HOG and color features.
After extracting HOG and color features, sklearn.preprocessing.StandardScaler() is used to normalize these extracted features. Then I train a linear SVM classifier to detect cars.
```python
X = np.vstack((car_features, notcar_features)).astype(np.float64)
X_scaler = StandardScaler()
X_scaler.fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
rand_state = np.random.randint(0, 100)
x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

svc = LinearSVC()
svc.fit(x_train, y_train)
print(svc.score(x_test, y_test))
```
### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window from 400 to 656 in y-axis at 1.5 scales all over the images and came up with this:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image8]
![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.
Here's a [link to my video result](./project_result.mp4)

#### 2. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

###  corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

Here is the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

In order to decrease false positives, I use a cache to average the collected boxes from previous frames. The method is `collections.deque`,  in this way I do not need to delete the oldest heatmap.

---

### Discussion

#### 1. .jpg and .png
The first problem I've meet in this project is the type of the image. The type of images in training set are .jpg and range from 0 to 255. While the test images and video frame are range from 0 ro 1. No car is detected in this condition, so I normalize the training set to [0, 1]. I've spent 2 days to find this bug and one line to fix it.

#### 2. Cars behind the fence
In the project_video.mp4, two cars(one white and one black) appearance on the forward road and some cars on the backward road at the other side of fence. In this case, there is no need to detect cars on the other side in most of time. However, those cars may cause serious traffic issue, so more viode should be used to test the pipeline in my opinion.

