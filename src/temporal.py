# src/temporal.py
import numpy as np

class Line:
    def __init__(self, alpha=0.1, max_frames_lost=15):
        self.detected = False  
        self.current_fit = None # Smoothed polynomial coeffs
        
        self.alpha = alpha # Smoothing factor
        
        # Confidence score [0.0 - 1.0]
        self.confidence = 0.0
        
        # Count of pixels in last good fit
        self.pixel_count = 0
        
        # How many frames since last good detection
        self.frames_since_detected = 0
        self.max_frames_lost = max_frames_lost # Frames to wait before triggering "TAKE OVER"

    def calculate_confidence(self, pixel_count):
        # A simple confidence score based on pixel count
        # min_pix (50) is 0%, 2000+ pixels is 100%
        min_p = 50
        max_p = 2000
        
        if pixel_count < min_p:
            return 0.0
            
        norm_pixels = (pixel_count - min_p) / (max_p - min_p)
        # Clip to be between 0.0 and 1.0
        return np.clip(norm_pixels, 0.0, 1.0)

    def update(self, new_fit_coeffs, pixel_count):
        """
        Updates the line's state based on a new (raw) fit.
        """
        if new_fit_coeffs is None:
            # New fit failed (e.g., sanity check)
            self.frames_since_detected += 1
            if self.frames_since_detected >= self.max_frames_lost:
                self.detected = False # Officially "lost" the line
            
            # Decay confidence
            self.confidence = max(0.0, self.confidence - 0.1) 
        else:
            # New fit is good
            self.frames_since_detected = 0
            self.detected = True
            self.pixel_count = pixel_count
            self.confidence = self.calculate_confidence(pixel_count)
            
            if self.current_fit is not None:
                # Apply exponential moving average
                self.current_fit = (self.alpha * new_fit_coeffs) + ((1 - self.alpha) * self.current_fit)
            else:
                # First detection
                self.current_fit = new_fit_coeffs