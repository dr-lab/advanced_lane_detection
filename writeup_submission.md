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

[corner1]: ./output_images/calibration/corner1.png "corner1 image"
[corner2]: ./output_images/calibration/corner2.png "corner2 image"
[corner3]: ./output_images/calibration/corner3.png "corner3 image"
[corner4]: ./output_images/calibration/corner4.png "corner4 image"

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

 ![gif][gif] 

### Camera Calibration

For camera calibration, first step is to compute the camera matrix and distortion coefficients. I tried following way to get the calibration parameters with CV2.

Bellow line is to get the corner points of each calibration images, 

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
Here are some samples we paint the corners, highlighted in red dots.
    ![corner1][corner1]
    ![corner2][corner2] 
    ![corner3][corner3] 
    ![corner4][corner4] 
    

Then calculate the matrix and distortion coefficiencies

    objpoints.append(objp)
    imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
In the return, we only need to use two parameters, mtx and dist. Others can be ignore in this project.

Following are some samples of the un-distorted images.

![calib1][calib1] 
![calib2][calib2] 
![calib3][calib3] 
![calib4][calib4] 
  
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 



### Pipeline for single image pre-processing

To demonstrate this step, I will describe how I apply the distortion correction to the test images like this one:
![pipeline1][pipeline1] 
![pipeline2][pipeline2] 
![pipeline3][pipeline3] 
![pipeline4][pipeline4] 

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
![birdview1][birdview1] 
![birdview2][birdview2] 
![birdview3][birdview3] 
![birdview4][birdview4] 

### Line Detection
Line detection is done on the un-distorted image, which also have perspective transformed. The logic of line detection is from the course video, 
It will find_lanes function will detect left and right lanes from the warped image and 'n' windows will be used to identify peaks of histograms.
 
 In this implementation, the histogram is playing a key role, the place of max value in the left side of the middle will be start point (x) of the left line, while the same for the right line.
 
 And a moving window is used to detect the next mean of the points (above a threshold), then use polyfit function to get the parameters of the poly line.
 
    left_fit = np.polyfit(left_lane_y, left_lane_x, 2)
 
At the end of the line painting, image will be transformed back to original perspective from the bird-view perspective. In the final image, we paint a green on the area between two detected lines such that the lane area is identified clearly.

![polyline1][polyline1] 
![polyline2][polyline2] 
![polyline3][polyline3] 
![polyline4][polyline4] 



### Master Function for Image Process
Bellow is the master Function which wrapper all the Functions we discussed above, and it will be passed to the video process step.

    def process_image(image):
        # Apply pipeline to the image to create black and white image
        img = pipeline(image)
        
        
        # Warp the image to make lanes parallel to each other
        top_down, perspective_M, perspective_Minv = corners_unwarp(img, mtx, dist)
        # Find the lines fitting to left and right lanes
        a, b, c, lx, ly, rx, ry, curvature = fit_lanes(top_down)
        # Return the original image with colored region
        return draw_poly(image, top_down, a, b, c, lx, ly, rx, ry, perspective_Minv, curvature)
        
        
 ### calculated the radius of curvature of the lane and the position of the vehicle with respect to center
        
The way to calculate the radius basically is bery roughly, and is way more from precision, but it do help giev some estimation of the curve. Bellow is the formula used 

First we define y-value where we want radius of curvature, for this I choose the maximum y-value, corresponding to the bottom of the image

    y_eval = np.max(yvals)
    
Then since our image is pixel based, so need to do some math on the pixels. I use following algorithm to transform pixel to meter.    
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    fit_cr = np.polyfit(yvals*ym_per_pix, fitx*xm_per_pix, 2)
    
Finally is the formula to calculate the curvature.    
    
    curverad = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*fit_cr[0])
        
When we do car position calculation, we use the image size which is pixel based. See bellow we use some hard coded pixel size to decide left or right in the lane. 
 
    position = image_shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension    
    return (position - center)*xm_per_pix  
          
          
**Note:** One thing is very important, that the calculation need to based on the original image perspective, not the bird view perspective.          
        

### Final Video Processing and Generation

I use the same way used in the P1 to process the video frame by frame, and then write to a new video mp4 file.

The final video can be found from [video](./project_video_output.mp4)



---

### Discussion

1. In this project, I use sobel which is different than the one used in P1, but same technologies like grey, GausioanBlur, are used in the pre-steps to pre-process images.
2. There are more other ways used here than P1, like camera calibration, perspective transformation, poly line fit.
 3. For next step, need more time to tune the line detection algorithms, to work on the challenge videos. 
 4. Performance wise, there should still some space which can be improved, e.g. re-use previous frame's finding, not full-scan the each frame.

  
