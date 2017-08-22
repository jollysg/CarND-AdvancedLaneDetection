
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/rectificationDemonstration.png "Undistorted"
[image2]: ./output_images/imageRectificationExample.png "Road Transformed"
[image3]: ./output_images/differentThresholdsComparison.png "Threshold comparison"
[image4]: ./output_images/thresholdedImageExample.png "Threshold Example"
[image5]: ./output_images/wawrpingImages.png "Warping of image"
[image6]: ./output_images/curveFit.png "Curve Fitting"
[image7]: ./output_images/laneDetectionDemonstration.png "Lane detection from a road image"
[image8]: ./output_images/laneDetectionExample.png "Lane Detection Result"
[video1]: ./output_images/project_video_out1.mp4 "Video"
[video2]: ./output_images/challenge_video_out2.mp4 "Video"
[video3]: ./output_images/harder_challenge_video_out2.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the lines #12 to #54 of the file called `advLaneDetect.py`. A class called CameraSetup was created that contains two methods - cameraCalibration and rectifyImage. The class uses opencv methods to calibrate and rectify the images provided to it. 'cameraCalibration' method takes the path of all the images that serve as an input for the calibration (ideally chess board images) and the number of corners for the chess board. It then uses the findChessboardCorners method in the openCV to find out the 'object points'. The "object points" will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These coefficients are stored as the member variables of the CameraSetup class so that they can be used at a later time to rectify images. Ideally, the cameraCalibration method of the class will be called only once during the initialization of the program. 

Once initialized, the 'rectifyImage' method of the CameraSetup class can be used to rectify distorted images received from the camera. This menthod uses the Open CV method `cv2.undistort()` to rectify the raw images. The distortion coeffiicient required by this openCV function are provided by the member variables of the CameraSetup class stored after the calibration. This method will always be called after the cameraCalibration method, and is called for processing each framed received from the camera. Following are the results: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #451 through #460 in `advLaneDetect.py`). Following image shows a comparison of different thresholding shemes evaluated. As can be seen, R channel thresholding was able to separate the lanes (both yellow and white) well. S Channel performed fine with the images, but performed poorly with the discontinuous lines at a farther distance and also the upper half of the image. This upper part however would have been masked by the region of interest selected at a later stage in the pipeline. Sobel X did a decent job in detecting the dotted lines farther away from the camera. The Sobel Y and the Magnitude Threshold were not taken due to horizontal lane lines detected by them. The directional threshold (although not shown in this image) was pretty noisy. 

![alt text][image3]

Hence the final thresholding used was a combination of r-channel as well as sobel X threshold. Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in a function called `advLaneDetectionPipeline()`, which appears in lines 332 through 347 in the file `advLaneDetect.py`.  The stps use open CV methods getPerspectiveTransform to get the perspective and inverse perspective transform using the source and destination vertices.  I chose the hardcode the source and destination points as follows:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 125, 720      | 125, 720        | 
| 565, 460      | 125, 0      |
| 715, 460     | 1230, 0      |
| 1230, 720      | 1230, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After obtaining the lane-line pixels from the step above, the resulting warped image was sliced vertically into 9 bars. Using a histogram, the approximate mid-point of both the lanes for each slices were determined. And then good pixels in a margin of 100 pixels on left and right of this point as well as 50 pixels above and below this point, were scanned. After this using the 2nd order polynomial fitting for all the resulting left and right points, the lane line was estimated. Polyfit method of the numpy library was used to find the 2nd order polynomial coefficients. This can be shown in the image below. The code for this can be found in the slidingWindows and skipSlidingWindows methods in the file advLaneDetect.py. 
![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the line 386 to 410 in the advLaneDetectionPipeline method in `advLaneDetect.py`. After finding the polynomial coefficients for the lane lines, the next step was to transform these points into the real world estimates. For this, following conversion was used from pixel to real world conversion: 25/720 meters per pixel in y dimension and 3.7/700 meters per pixel in the x dimension of the image. Since the camera is pointing forward and downward with a small angle in front of the car, the vertical pixels represent the distance of the objects, with the upper portion of the image further away from the car. Following calculations were used to find the curvature of the left and right lanes:

```python
    ym_per_pix = 25 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftFitX * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightFitX * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
```
To find the vehicle position in the lane, I assumed that the camera is mounted at the center of the vehicle. With that, the center of the image will coincide with the center of the lane, if the vehicle is in the center. Hence, at the closes point (i.e. at the bottom of the image), from the fitted left and right curves, the center of the lanes was estimated using (leftFitPoint + rightFitPoint)/2. The leftFitPoint and rightFitPoint were chosen at y=719 (i.e. bottom of the image). The offset in pixels is the difference between the center pixels of the image and the center pixels of the lane.  These pixels are converted into real world coordinates in meters using the conversion factor for the x-dimension discussed in the first para of this section.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The resultant points and the polynomial line from the step above are used to draw a polygon highlighting the lane for the vehicle. This is then warped back into the image using the inverse perspective transform. This can be seen from the line 380 to 383 of the advLaneDetectionPipile function in the advLaneDetect.py. The polygon is then drawn over the camera image so as to show the identified lane on the road. This can be seen in the line 516, function process_image of the file advLaneDetect.py. Following is an example image of the result.  

![alt text][image7]

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

##### Thresholding
AS mentioned above, I used a combination of R-channel and Sobel X thresholding. This worked fine for the project_video, however, there are quite a lot of portions in the challenge_video.mp4 and the harder_challenge_video.mp4 where this implementation fails. In the challenge video, there are following places where this fails. Firstly, the lane has two colors, and the thresholding algorithm detects the change in the color as a line. As this patch line between the two color moves closer to the lane lines, the detection begins to fail. Secondly, the divider of the road casts a shadow and there are sections of the highway where these shadows are close to the yellow lane lines, which means they are in the region of interest. Thirdly, when the car goes under the bridge, the shadow of the bridge messes up the lane detection. In a similar way, there are portions in the harder_challenge_video.mp4 where the road is too bright as the car goes in between the shades. All these problems are related to thresholding. A better thresholding technique needs to be explored for that.

##### Curvature of the road
This is a problem seen more in the harder_challenge video. The road curvature at many portions of this video is too high for the algorithm to handle. As a result one of the lanes goes out of the region of interest. For that the region of interest needs to be tweaked so that both the lanes are visible in the perspective transform. In extreme cases, the algorithm may have to make some assumptions about the estimated position of the road line that is not visible in the camera.

##### Skipping of sliding windows for detecting lane lines
In the implementation, the algorithm runs a sliding window initially to identify the lane lines. Once the lines are detected, it skips the sliding window once the position of the lanes are identified. However, there are portions of both the challenge videos where the algorithm is not able to detect any lanes due to the reasons cited above. As a result, the algorithm should fall back to finding the lanes using the sliding windows for the next frames. However, the current check for whether the sliding window should be used or skipped is very weak. As a result, once the algorithm is not able to detect any lanes in any of the frames, the detection in the subsequent frames fails. This check was happening in the line 366 to 369 of the advLaneDetectionPipeline function in the advLaneDectection.py file. However, due to this problem, this check has been disabled and the algorithm carries out the sliding window search in all the frames. A better check needs to be implemented over here. 

Following are links to the performance of the algorithm on the challenge videos:

[link to the challenge video output][video2]
 
[link to the harder challenge video output][video3]
 
