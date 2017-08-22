import numpy as np
import cv2
import glob
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
from moviepy.editor import *
#%matplotlib qt

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d points in real world space
# imgpoints = [] # 2d points in image plane.
#
# # Make a list of calibration images
# images = glob.glob('camera_cal/calibration*.jpg')
#
# print(len(images))
#
# # Step through the list and search for chessboard corners
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
#     # Find the chessboard corners
#     ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
#
#     # If found, add object points, image points
#     if ret == True:
#         objpoints.append(objp)
#         imgpoints.append(corners)
#
#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)
#
# cv2.destroyAllWindows()

class CameraSetup:


    def cameraCalibration(self, imagePaths, size=(9, 6)):
        nc = size[0]
        nr = size[1]
        objp = np.zeros((nr * nc, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nc, 0:nr].T.reshape(-1, 2)
        print('length of images: ', len(imagePaths))

        objpoints = []
        imgpoints = []
        gray = None
        # Step through the list and search for chessboard corners
        for fname in imagePaths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
        #         img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        #         cv2.imshow('img', img)
        #         cv2.waitKey(500)
        # cv2.destroyAllWindows()

        camMat, distCoeffs = None, None

        if gray != None:
            retVal, self.camMat, self.distCoeffs, self.rotVecs, self.transformVecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return camMat, distCoeffs

    def rectifyImage(self, srcImage):
        undistImage = cv2.undistort(srcImage, self.camMat, self.distCoeffs)

        return undistImage

class ImageProc:
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # Apply threshold
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        # 3) Take the absolute value of the derivative or gradient
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        # 6) Return this mask as your binary_output image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel = None
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        # Apply threshold
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Calculate the magnitude
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        # 5) Create a binary mask where mag thresholds are met
        # 6) Return this mask as your binary_output image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.sqrt(np.square(sobelx) + np.square(sobely))
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        mag_binary = np.zeros_like(scaled_sobel)
        thresh_min = mag_thresh[0]
        thresh_max = mag_thresh[1]
        mag_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return mag_binary

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        # Apply threshold
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        #    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        dir_binary = np.zeros_like(grad_dir)
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        dir_binary[(grad_dir >= thresh_min) & (grad_dir <= thresh_max)] = 1
        return dir_binary

    def grayThreshold(self, rgbImage, thresh = (0, 255)):
        gray = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2GRAY)
        binary = np.zeros_like(gray)
        binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
        return binary

    def colorThreshold(self, rgbImage, color = 'r', thresh = (0, 255)):
        colorChannel = None
        if color == 'r':
            colorChannel = rgbImage[:, :, 0]
        elif color == 'g':
            colorChannel = rgbImage[:, :, 1]
        else:   # blue
            colorChannel = rgbImage[:, :, 2]
        binary = np.zeros_like(colorChannel)
        binary[(colorChannel > thresh[0]) & (colorChannel <= thresh[1])] = 1
        return binary

    def hlsThreshold(self, rgbImage, color = 's', thresh = (0, 255)):
        hlsImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HLS)
        hlsChannel = None
        if color == 'h':
            hlsChannel = hlsImage[:, :, 0]
            if thresh[1] > 179:
                thresh = (thresh[0], 179)
        elif color == 'l':
            hlsChannel = hlsImage[:, :, 1]
        else:   # s channel
            hlsChannel = hlsImage[:, :, 2]
        binary = np.zeros_like(hlsChannel)
        binary[(hlsChannel > thresh[0]) & (hlsChannel <= thresh[1])] = 1
        return binary



def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

left_fit = None
right_fit = None
def slidingWindows(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
#    out_img = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2RGB)
#    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # plt.figure(figsize=(10, 5))
    # plt.imshow(out_img)
    # plt.show()

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    global left_fit, right_fit
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # print(out_img.shape)
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.figure(figsize=(10, 5))
    # plt.imshow(out_img)
    # # plt.plot(leftx, lefty, color='red')
    # plt.plot(rightx, righty, color='blue')
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    # y_eval = np.max(ploty)
    # # Define conversions in x and y from pixels space to meters
    # ym_per_pix = 30 / 720  # meters per pixel in y dimension
    # xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    #
    # # Fit new polynomials to x,y in world space
    # left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    # right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # # Calculate the new radii of curvature
    # left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    #     2 * left_fit_cr[0])
    # right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    #     2 * right_fit_cr[0])
    # # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # # Example values: 632.1 m    626.2 m

    return  left_fitx, right_fitx, ploty

## Taken from Udacity CarND tutorials
def skipSlidingWindow(binary_warped):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    global left_fit, right_fit
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

def advLaneDetectionPipeline(inputImage):
    ysize = inputImage.shape[0]
    xsize = inputImage.shape[1]

    vertices = np.array([[(125, ysize), (xsize/2-75, ysize/2+100), (xsize/2 + 75, ysize/2+100),(xsize - 50, ysize)]], dtype = np.int32)
    destVertices = np.array([[(125, ysize), (125, 0), (xsize-50, 0),(xsize - 50, ysize)]], dtype = np.float32)

    maskedImage = region_of_interest(inputImage, vertices=vertices)
    drawnImage = inputImage.copy()
    cv2.polylines(drawnImage, pts=vertices, isClosed=True, color=(255, 0, 0), thickness=2)

    M = cv2.getPerspectiveTransform(np.float32(vertices), destVertices)
    Minv = cv2.getPerspectiveTransform(destVertices, np.float32(vertices))
    warped = cv2.warpPerspective(inputImage, M, (drawnImage.shape[1], drawnImage.shape[0]))

    print(np.max(warped))
    binary_warped = np.zeros_like(warped, dtype=np.uint8)
    binary_warped[warped != 0] = 1

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(drawnImage, 'gray')
    plt.xlabel('original image')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_warped, 'gray')
    plt.xlabel('warped image')

    plt.show()

    if left_fit == None or right_fit == None:
        leftFitX, rightFitX, ploty = slidingWindows(binary_warped)
    else:
        leftFitX, rightFitX, ploty = skipSlidingWindow(binary_warped)

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([leftFitX, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rightFitX, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (inputImage.shape[1], inputImage.shape[0]))
    # Combine the result with the original image

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
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
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    radius = np.mean([left_curverad, right_curverad])
    cv2.putText(newwarp, 'Lane Radius: {}m'.format(int(radius)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)

    laneCenter = (leftFitX[ysize-1] + rightFitX[ysize-1]) / 2.
    offsetInPixels = xsize/2. - laneCenter
    offsetInMeters = offsetInPixels * xm_per_pix
    # print('lane offset is ', offsetInMeters)
    cv2.putText(newwarp, 'Lane Offset: {:.2f}m'.format(offsetInMeters), (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, 255)


    return newwarp

def playVideo(path):
    imageio.plugins.ffmpeg.download()

    clip = VideoFileClip(path)
    x = clip.iter_frames(dtype="uint8")
    print(clip.duration)

    t = 0.
    while (t < clip.duration):

        frame = clip.get_frame(t)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('window', bgr)
        cv2.waitKey(40)
        t = t + 0.04

    cv2.destroyAllWindows()


def main(args):
    cs = CameraSetup()
    calibrationImageFolder = 'camera_cal'
    print(calibrationImageFolder + '/*.jpg')
    imagePaths = glob.glob(calibrationImageFolder + '/*.jpg')
    print(imagePaths)
    cs.cameraCalibration(imagePaths, (9, 6))
    print('Destroying all the windows...')
#    cv2.destroyAllWindows()

    test_image_path = 'test_images/straight_lines1.jpg'

#    playVideo('project_video.mp4')

    clip = VideoFileClip('project_video.mp4')
    import time
    t = time.time()
    tframe = time.time() - t
    ip = ImageProc()

    # plt.figure(figsize=(5, 5))
   # plt.ion()


    while (tframe < clip.duration):

        tframe = time.time() - t
        frame = clip.get_frame(tframe)

        image = cs.rectifyImage(frame)
        # bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow('window', bgr)
        # cv2.waitKey(1)

        rChannel = ip.colorThreshold(image, 'r', (200, 255))
        sChannel = ip.hlsThreshold(image, 's', (90, 255))

        sobelxImage = ip.abs_sobel_thresh(image, thresh=(20, 100))
        # sobelyImage = ip.abs_sobel_thresh(image,orient='y', thresh = (20, 100))
        # magThreshImage = ip.mag_thresh(image, mag_thresh=(20, 100))
        dirThreshImage = ip.dir_threshold(image, sobel_kernel=15, thresh=(0.9, 1.1))

        combined_GrayScale = np.zeros_like(dirThreshImage)
        combined_GrayScale[(sobelxImage == 1) | (sChannel == 1)] = 255

        # combined_GrayScale = np.zeros_like(combined)
        # combined_GrayScale[combined == 1] = 255

        newwarp = advLaneDetectionPipeline(combined_GrayScale)

        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        # plt.imshow(result)
        # plt.show()

        # plt.figure(figsize=(10, 5))
        # plt.imshow(result)
        # plt.xlabel('lanes')
        # plt.title('lanes detected plot')
        #
        # plt.show()

        bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow('detected lanes', bgr)
        cv2.waitKey(1)


    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
