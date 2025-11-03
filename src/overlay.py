# src/overlay.py
import cv2
import numpy as np

def draw_lane_overlay(original_image, Minv, left_line, right_line, lat_offset_m, avg_curve_rad_m):
    """
    Draws the detected lane polygon and the full HUD with metrics.
    """
    
    # Create a blank image to draw the lane polygon on
    warp_zero = np.zeros_like(original_image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, original_image.shape[0]-1, original_image.shape[0])
    
    # Check if lines were continuously detected
    if left_line.detected and right_line.detected:
        left_fit = left_line.current_fit
        right_fit = right_line.current_fit
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane polygon (green)
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    
    # Warp the polygon back to original image space
    new_warp = cv2.warpPerspective(color_warp, Minv, 
                                   (original_image.shape[1], original_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, new_warp, 0.3, 0)
    
    # --- Draw the Full HUD ---
    # Detection Status
    left_status = "YES" if left_line.detected else "NO"
    right_status = "YES" if right_line.detected else "NO"
    left_conf_str = f"{left_line.confidence:.2f}"
    right_conf_str = f"{right_line.confidence:.2f}"
    
    cv2.putText(result, f"Left: {left_status} (Conf: {left_conf_str})", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f"Right: {right_status} (Conf: {right_conf_str})", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Format offset text (positive=Left, negative=Right)
    offset_dir = "L" if lat_offset_m > 0 else "R"
    cv2.putText(result, f"Offset: {abs(lat_offset_m):.2f}m {offset_dir}", (50, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- TAKE OVER Warning ---
    if not left_line.detected or not right_line.detected:
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 250), (original_image.shape[1], 350), (255, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        
        cv2.putText(result, "DRIVER TAKE OVER", (int(original_image.shape[1] / 2) - 300, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return result