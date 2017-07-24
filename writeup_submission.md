## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[gif]: ./project_video_output.gif
[calib1]: ./output_images/calibration/1.png "calib1 image"
[calib2]: ./output_images/calibration/2.png "calib2 image"
[calib3]: ./output_images/calibration/3.png "calib3 image"
[calib4]: ./output_images/calibration/4.png "calib4 image"

[pipeline1]: ./output_images/pipeline/1.png "pipeline1 image"
[pipeline2]: ./output_images/pipeline/2.png "pipeline2 image"
[pipeline3]: ./output_images/pipeline/3.png "pipeline3 image"
[pipeline4]: ./output_images/pipeline/4.png "pipeline4 image"

[polyline1]: ./output_images/poly_lanes/poly_lane1.png "polyline1 image"
[polyline2]: ./output_images/poly_lanes/poly_lane2.png "polyline2 image"
[polyline3]: ./output_images/poly_lanes/poly_lane3.png "polyline3 image"
[polyline4]: ./output_images/poly_lanes/poly_lane4.png "polyline4 image"


[birdview1]: ./output_images/warp/1.png "bird view1 image"
[birdview2]: ./output_images/warp/2.png "bird view2 image"
[birdview3]: ./output_images/warp/3.png "bird view3 image"
[birdview4]: ./output_images/warp/4.png "bird view4 image"


[video1]: ./project_video.mp4 "Video"
[video_output]: ./project_video_output.mp4.mp4 "Video"

Bellow is the lane detection video output (in gif ), video can be downloaded in the root folder in the repo.

 [gif][gif] 

### Camera Calibration

For camera calibration, first step is to compute the camera matrix and distortion coefficients. I tried following way to get the calibration parameters with CV2.

Bellow line is to get the corner points of each calibration images, 

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

Then calculate the matrix and distortion coefficiencies

    objpoints.append(objp)
    imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
In the return, we only need to use two parameters, mtx and dist. Others can be ignore in this project.

Following are some samples of the un-distorted images.

|![calib1][calib1] |
|![calib2][calib2] |
|![calib3][calib3] |
|![calib4][calib4] |
  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 



### Pipeline for single image pre-processing

To demonstrate this step, I will describe how I apply the distortion correction to the test images like this one:
|![pipeline1][pipeline1] |
|![pipeline2][pipeline2] |
|![pipeline3][pipeline3] |
|![pipeline4][pipeline4] |

First step is do a Gaussian Blur on the image

    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
Then convert to HLS color space and separate the S channel per session video
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    
Also convert to gray image
    
    # Grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
Finally apply the sobel function for both X and Y, and combine them together
    
    # Define sobel kernel size
    ksize = 7
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(60, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(40, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(.65, 1.05))
    # Combine all the thresholding information
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
At the end, one more step  which helps the image process time, is to crop the image of "area of interested"

    color_binary = region_of_interest(color_binary, vertices)

When process the image, we also add one step to transform the image, which is to un-distort the image based on the camera calibration parameters, matrix and distortion coefficiencies.
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
Then use the cv2 transform the perspective
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)
    
To visualize the perspective transformation clear, bellow is one sample which do the transformation on the original color image. 
 (In our implementation, this step done on the grey image at the end of the image pre-processing pipeline)
|![birdview1][birdview1] |
|![birdview2][birdview2] |
|![birdview3][birdview3] |
|![birdview4][birdview4] |

### Line Detection
Line detection is done on the un-distorted image, which also have perspective transformed. 


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
