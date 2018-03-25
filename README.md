**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
*
# How I complete the pipeline
I complete this pipeline by threes steps:
#### The first step: Initial completion of pipeline
>* 1. Load traing data using `glob`, the data include cars and noncars.
>* 2. Show some images.
>* 3. Use the [scikit-image](http://scikit-image.org/) to extract Histogram of Oriented Gradient features. The documentation for this function can be found [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) and a brief explanation of the algorithm and tutorial can be found [here](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html).And then test it.
>* 4. Use `np.histogram` to get histograms of color, and then test it.
>* 5. Get the spatially binned features, and then test it.
>* 6. Define a `single_img_features` to get the single image's features, and can choose as: `spatial_feat=True, hist_feat=True, hog_feat=True`
>* 7. Define a function `extract_features` to extract features from a list of images, the function `extract_features` use the `single_img_features` which defined before.
>* 8. Define the parameters used in train a classifier Linear SVM classifier.
>* 9. Train a classifier Linear SVM classifier.
>* 10. Implement a sliding-window, 
>* 11. Explore a more efficient method for doing the sliding window approach, one that allows us to only have to extract the Hog features once. This method is defined in `find_cars`, and in which We should choose the same parameters used in train a classifier Linear SVM classifier, such as `cspace,hog_channel, spatial_feat, hist_feat, hog_feat` and others.
>* 12. Create a heat map and reject outliers.
>* 13. Estimate a bounding box for vehicles detected  use `label`. And show them.
>* 14. Finish the pipeline.
#### The second step: Parameter Tuning
it is a 
>* 1. Test on the test images.
>* 2. Change the parameters, such as `color_space,hog_channel, spatial_feat, hist_feat, hog_feat...`and **multi-scale Windows**.
>* 3. Do as code:
```pyton
if car detection is fine:
        go to third step
    else:
        go to second step
```
#### The third step: 
>* 1. Run pipeline on a video stream(both on test_video.mp4 and project_video.mp4)

[//]: # (Image References)
[image1]: ./examples/origin_img.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 
You're reading it! Submit my writeup as markdown

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
Using the [scikit-image](http://scikit-image.org/) to extract Histogram of Oriented Gradient features. The documentation for this function can be found [here](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=feature%20hog#skimage.feature.hog) and a brief explanation of the algorithm and tutorial can be found [here](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html).
The code for this is :
```python
def get_hog_features(img,orient=9, pix_per_cell=8, cell_per_block=2
                    ,vis=False,feature_vec=True):
    """
    return the HOG feature extraction and hog images.
    """
    if vis==True:
        hog_features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  visualise=vis, feature_vector=feature_vec,
                                  transform_sqrt=False, 
                                  block_norm="L2-Hys")
        return hog_features, hog_image
    else: 
        hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell), 
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  visualise=vis, feature_vector=feature_vec,
                                  transform_sqrt=False, 
                                  block_norm="L2-Hys")
        return hog_image 
```

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

After I tried many combinations of parameters, I choose the the parameters which had highest test Accuracy in SVM classifier. Finally the parameters are as below:


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

