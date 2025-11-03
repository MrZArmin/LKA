# src/preprocess.py
import cv2
import numpy as np

def preprocess_image(image):
    # Convert to HLS.
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]  # S channel (Saturation)
    l_channel = hls[:, :, 1]  # L channel (Lightness)
    
    # 1. S-channel thresholding (for color)
    s_thresh_min = 100
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # 2. L-channel thresholding (for brightness)
    # Lane lines are bright, asphalt edges are not
    l_thresh_min = 120
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
    
    # 3. Sobel X on L-channel (for vertical edges)
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
    
    sx_thresh_min = 20
    sx_thresh_max = 120
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1
    
    # 4. Combine the masks
    # (Color OR (Edge AND Bright))
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | ((sx_binary == 1) & (l_binary == 1))] = 1
    
    return combined_binary