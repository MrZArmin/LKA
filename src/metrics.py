# src/metrics.py
import numpy as np

def define_metrics(img_size, warped_width_px):
    """
    Define conversion factors from pixels to meters.
    We calibrate this based on the warped image.
    """
    width, height = img_size
    
    # Standard U.S. lane width is 3.7 meters
    # In our warped image, the lane width is (width - 2 * offset), 
    # which is the 'warped_width_px'
    xm_per_pix = 3.7 / warped_width_px
    
    # A common assumption is that the dashed lines are ~3m long
    # and we see ~30m down the road.
    ym_per_pix = 30 / height 
    
    return xm_per_pix, ym_per_pix

def calculate_curvature_m(left_fit, right_fit, ploty, xm_per_pix, ym_per_pix):
    """
    Calculates the radius of curvature in meters.
    """
    if left_fit is None or right_fit is None:
        return 0, 0 # Return 0 if fit failed
        
    # We'll measure curvature at the bottom of the image (closest to the car)
    y_eval = np.max(ploty) * ym_per_pix # y-position in meters

    # Re-fit polynomials to x,y in meters
    left_fit_cr = np.polyfit(ploty * ym_per_pix, 
                             (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]) * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, 
                              (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]) * xm_per_pix, 2)

    # Calculate radius using the formula: R = (1 + (2Ay + B)^2)^(3/2) / |2A|
    A_left = left_fit_cr[0]
    B_left = left_fit_cr[1]
    left_curve_rad = ((1 + (2*A_left*y_eval + B_left)**2)**1.5) / np.absolute(2*A_left)
    
    A_right = right_fit_cr[0]
    B_right = right_fit_cr[1]
    right_curve_rad = ((1 + (2*A_right*y_eval + B_right)**2)**1.5) / np.absolute(2*A_right)
    
    return left_curve_rad, right_curve_rad

def calculate_offset_m(left_fit, right_fit, img_width, xm_per_pix):
    """
    Calculates the car's lateral offset from the lane center in meters.
    """
    if left_fit is None or right_fit is None:
        return 0.0 # No fit, assume center
        
    # Y-position to measure at (bottom of the image)
    y_eval = 719 # Image height - 1
    
    # Find x position of left and right lanes at the bottom
    left_x_px = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x_px = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    # Find lane center in pixels
    lane_center_px = (left_x_px + right_x_px) / 2
    
    # Find car center in pixels
    car_center_px = img_width / 2
    
    # Calculate offset in pixels
    offset_px = car_center_px - lane_center_px
    
    # Convert to meters
    offset_m = offset_px * xm_per_pix
    
    return offset_m