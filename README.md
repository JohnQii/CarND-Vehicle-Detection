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
[image1]: ./output_images/origin_img.png
[image2]: ./output_images/HOG_example.png
[image21]: ./output_images/color_hist.png
[image22]: ./output_images/bin_spatial.png
[image3]: ./output_images/350-500windows.png
[image4]: ./output_images/400-650windows.png
[image5]: ./output_images/multi-windows.png
[image6]: ./output_images/heatall.png
[video1]: ./project_video-out.mp4
[video2]: ./test_video-out.mp4
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
Here is an example of hog feature:
![alt text][image2]
Here is an example of color space hist feature:
![alt text][image21]
Here is an example of spatital bin feature:
![alt text][image22]
#### 2. Explain how you settled on your final choice of HOG parameters.

After I tried many combinations of parameters, I choose the the parameters which had highest test Accuracy in SVM classifier：
| hog_feat | spatial_feat | hist_feat | color_sapce | orient | pix_per_cell | cell_per_block | hog_channel | Test Accuracy |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|True| False | False |YUV |9 |16 |2 |ALL |0.9752|
| True | False | False |YUV |11 |16| 2 |ALL |0.9789|
| True | True | True |YUV |11| 16| 2 |ALL |0.9724|
| True | True | True |YUV |11| 8 |2 |ALL |0.9769|
| True | True | True |YUV |10 |8| 2 |ALL |0.9828|
| True | True | True |YCrCb |10| 8 |2| ALL |0.9840|
| True | True | True| YCrCb |11| 8 |2| ALL| 0.9901|

Finally the parameters are as below:
| hog_feat | spatial_feat | hist_feat | color_sapce | orient | pix_per_cell | cell_per_block | hog_channel | Test Accuracy |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| True | True | True| YCrCb |11| 8 |2| ALL| 0.9901|

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

* I trained a linear SVM using hog features, spatial features and hist features, The code is in `vehicle_detection.ipynb`
* extract features in function `def extract_features`
* And divide the test and train data by 0.2.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I don't know what size the car will be, and also where the car in images. So I do some test on the size of car in images, I get that the car which far from camera, it will be small size. So I use the two size sliding windows:
1. y 350 to 500 with scaling factor 1:
![alt text][image3]
2. y 400 to 656 with scaling factor 1.5:
![alt text][image4]
3. Combine the two windows.
![alt text][image5]
#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
The follow imags can show how my pipeline is working:

1. I have done many experiments and choose the best paras.
2. I use multi-windows which can improve the performance.
3. I use three features: hog features, spatial features and hist features,
---

### Video Implementation

#### 1. Provide a link to your final video output. 
Here's a [link to my test video result](./test_video-out.mp4) 
Here's a [link to my project video result](./project_video-out.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

1. I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
here is images to show this flow:
![alt text][image6]

and here is final code to process image:
```python
def process_image(img):
    boxes_list=[]
    
    ystart = 380
    ystop = 500
    scale = 1
    out_img, boxes = find_cars(img, ystart, ystop, scale, colorspace, hog_channel,
                        svc, X_scaler,
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                       color_feat)
    boxes_list.append(boxes)
    
    ystart = 400
    ystop = 656
    scale = 1.5
    
    out_img, boxes = find_cars(img, ystart, ystop, scale, colorspace, hog_channel,
                        svc, X_scaler,
                        orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
                       color_feat)
    boxes_list.append(boxes)
    
    boxes_list = [item for sublist in boxes_list for item in sublist]
    # make a heat-map 
    heat_img = np.zeros_like(img[:,:,0]).astype(np.float)
    heat_img = add_heat(heat_img, boxes)
    heat_img = apply_threshold(heat_img, 2)
    
    #Using label to find the box.
    labels = label(heat_img)    
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img
```
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. I think we should use some filtering algorithm to connect objects frames in videos.such as Karman filters.
2. Traing data can not be large enough, on the one hand, we can collect more data, on the other hand, we can combine the traditional detection algorithm with SVM!

