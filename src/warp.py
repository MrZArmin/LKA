# src/warp.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_user_warp_points(frame):
    """
    Displays the frame and waits for the user to click 4 points.
    Returns the selected source points.
    """
    # Convert to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print("--- Perspective Calibration ---")
    print("Please click 4 points on the image in this order:")
    print("  1. Top-Left corner of the lane trapezoid")
    print("  2. Top-Right corner")
    print("  3. Bottom-Right corner")
    print("  4. Bottom-Left corner")
    print("Close the pop-up window when done.")
    
    fig = plt.figure()
    plt.imshow(frame_rgb)
    plt.title("Click 4 points: TL, TR, BR, BL")
    
    # Wait for 4 mouse clicks
    points = plt.ginput(4, timeout=0)
    plt.close(fig)
    
    print(f"Points selected: {points}")
    # Order is [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    return np.float32(points)

def get_warp_matrices(img_size, src_points):
    """
    Calculates M and Minv from source points.
    """
    width, height = img_size
    
    # Source points (from user)
    # Order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    src = src_points 
    
    # Destination points (the rectangle we want to warp to)
    offset = 300 # Margin from the sides
    dst = np.float32([
        [offset, 0],                # Top-Left
        [width - offset, 0],        # Top-Right
        [width - offset, height],   # Bottom-Right
        [offset, height]            # Bottom-Left
    ])
    
    # Calculate the transform matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    # Calculate the inverse matrix Minv (for drawing later)
    Minv = cv2.getPerspectiveTransform(dst, src) 
    
    return M, Minv

def warp_image(image, M):
    """
    Applies the perspective warp using the matrix M.
    """
    img_size = (image.shape[1], image.shape[0])
    # Warp the image to a top-down ("bird's-eye") view
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped