# src/lane_fit.py
import numpy as np
import cv2

def histogram(image):
    # Get the bottom half of the image
    bottom_half = image[image.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    
    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0]//2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    
    return left_x_base, right_x_base

def sliding_window_search(binary_warped, left_x_base, right_x_base):
    # Create an output image to draw on
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    n_windows = 9
    margin = 100
    min_pix = 50

    window_height = int(binary_warped.shape[0] // n_windows)

    # Find the x and y coordinates of all white pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_x_current = left_x_base
    right_x_current = right_x_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        win_xleft_low = left_x_current - margin
        win_xleft_high = left_x_current + margin
        win_xright_low = right_x_current - margin
        win_xright_high = right_x_current + margin

        # Draw the windows (for debugging)
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > min_pix:
            left_x_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_pix:
            right_x_current = int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass # Handle empty arrays

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Color the found pixels
    out_img[lefty, leftx] = [255, 0, 0] # Red
    out_img[righty, rightx] = [0, 0, 255] # Blue

    return leftx, lefty, rightx, righty, out_img # Return debug image

def fit_polynomial(leftx, lefty, rightx, righty):
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
    except (np.linalg.LinAlgError, TypeError, ValueError):
        left_fit = None 

    try:
        right_fit = np.polyfit(righty, rightx, 2)
    except (np.linalg.LinAlgError, TypeError, ValueError):
        right_fit = None

    return left_fit, right_fit

def sanity_check(left_fit, right_fit, img_height):
    """
    Checks if the detected lines are plausible.
    Returns (left_fit, right_fit) which may be None if checks fail.
    """
    # Check 0: Did the polynomial fit succeed?
    if left_fit is None or right_fit is None:
        return None, None # failed

    # Check 1: Are lines roughly parallel?
    # Compare their curvature (the 'A' coefficient)
    curve_diff = abs(left_fit[0] - right_fit[0])
    if curve_diff > 0.001: # Threshold
        return None, None # Not parallel, FAIL and return
        
    # Check 2: Are they the right distance apart?
    # Check at the bottom of the image (y = img_height)
    y_eval = img_height - 1
    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    distance = right_x - left_x
    # Plausible distance in our warped view (from offset=300)
    # Expected: 1280 - 600 = 680
    if not (500 < distance < 850): 
        return None, None # Wrong distance, FAIL and return
        
    # All checks passed!
    return left_fit, right_fit

def find_lane_fits(binary_warped):
    """
    Main function for this module.
    Returns: left_fit, right_fit, debug_image, left_pixel_count, right_pixel_count
    """
    
    left_x_base, right_x_base = histogram(binary_warped)
    
    leftx, lefty, rightx, righty, debug_img = sliding_window_search(
        binary_warped, left_x_base, right_x_base
    )
    
    # Get the raw polynomial fits
    left_fit_raw, right_fit_raw = fit_polynomial(leftx, lefty, rightx, righty)
    
    # Run sanity checks
    left_fit_checked, right_fit_checked = sanity_check(
        left_fit_raw, right_fit_raw, binary_warped.shape[0]
    )

    # --- Add the fitted lines to the debug image ---
    if left_fit_checked is not None:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            left_fitx = left_fit_checked[0]*ploty**2 + left_fit_checked[1]*ploty + left_fit_checked[2]
            for i in range(len(ploty)):
                cv2.circle(debug_img, (int(left_fitx[i]), int(ploty[i])), 2, (255, 255, 0), -1) # Yellow
        except TypeError:
            pass # Should not happen now, but good to keep

    if right_fit_checked is not None:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        try:
            right_fitx = right_fit_checked[0]*ploty**2 + right_fit_checked[1]*ploty + right_fit_checked[2]
            for i in range(len(ploty)):
                cv2.circle(debug_img, (int(right_fitx[i]), int(ploty[i])), 2, (0, 255, 255), -1) # Cyan
        except TypeError:
            pass

    return left_fit_checked, right_fit_checked, debug_img, len(leftx), len(rightx)