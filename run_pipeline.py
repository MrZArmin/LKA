# run_pipeline.py
import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import sys

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from preprocess import preprocess_image
    from warp import get_user_warp_points, get_warp_matrices, warp_image
    from lane_fit import find_lane_fits
    from overlay import draw_lane_overlay
    from temporal import Line
    from debug_utils import create_debug_collage
    from csv_writer import CSVWriter
    from metrics import define_metrics, calculate_curvature_m, calculate_offset_m # <--- NEW
except ImportError as e:
    print(f"Error: {e}")
    print("Please make sure all module files (preprocess.py, warp.py, etc.) are in the 'src' directory.")
    sys.exit(1)

IMAGE_EXT = ['.jpg', '.jpeg', '.png']
VIDEO_EXT = ['.mp4', '.avi', '.mov']

def process_frame(frame, M, Minv, left_line, right_line, xm_per_pix, ym_per_pix, get_debug=False):
    """
    Runs the full pipeline on one frame.
    """
    img_height, img_width = frame.shape[:2]
    ploty = np.linspace(0, img_height - 1, img_height)

    # 1. Preprocessing
    binary_mask = preprocess_image(frame)
    
    # 2. Perspective Transform
    warped_binary = warp_image(binary_mask, M)
    
    # 3. Find raw lane fits
    left_fit_raw, right_fit_raw, warped_debug_img, l_count, r_count = find_lane_fits(warped_binary)
    
    # 4. Update smoothers
    left_line.update(left_fit_raw, l_count)
    right_line.update(right_fit_raw, r_count)
    
    # 5. Calculate Metrics
    lat_offset_m = 0.0
    avg_curve_rad_m = 0.0
    
    if left_line.detected and right_line.detected:
        # Calculate offset
        lat_offset_m = calculate_offset_m(left_line.current_fit, right_line.current_fit, 
            img_width, xm_per_pix)
        
        # Calculate curvature
        left_curve, right_curve = calculate_curvature_m(
            left_line.current_fit, right_line.current_fit, 
            ploty, xm_per_pix, ym_per_pix
        )
        avg_curve_rad_m = (left_curve + right_curve) / 2
        
    # 6. Draw final overlay
    final_overlay = draw_lane_overlay(frame.copy(), Minv, left_line, right_line, 
        lat_offset_m, avg_curve_rad_m)
    
    if get_debug:
        debug_images = {
            'original': frame.copy(),
            'binary': binary_mask, 
            'warped': warped_debug_img,
            'final': final_overlay
        }
        return final_overlay, debug_images, left_line, right_line, lat_offset_m
    
    return final_overlay, None, left_line, right_line, lat_offset_m

def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_path.exists(): ...
    ext = input_path.suffix.lower(); is_video = ext in VIDEO_EXT; first_frame = None; cap = None
    if is_video:
        cap = cv2.VideoCapture(str(input_path)); ret, first_frame = cap.read()
    elif ext in IMAGE_EXT:
        first_frame = cv2.imread(str(input_path))
    else:
        print(f"Error: Unknown file type: {ext}"); return
    if first_frame is None:
        print("Error: Could not load the first frame.");
        if cap: cap.release(); return

    img_size = (first_frame.shape[1], first_frame.shape[0])
    
    print("Waiting for user calibration...")
    src_points = get_user_warp_points(first_frame)
    
    # Define metrics based on calibration
    # We pass the 'offset' (300) from warp.py * 2
    warped_lane_width_px = img_size[0] - (300 * 2) 
    xm_per_pix, ym_per_pix = define_metrics(img_size, warped_lane_width_px)
    
    M, Minv = get_warp_matrices(img_size, src_points)
    print("Calibration complete. Processing...")
    
    left_line = Line(alpha=0.1)
    right_line = Line(alpha=0.1)
    
    csv_path = output_dir / f"{input_path.stem}_per_frame.csv"
    headers = ["frame_id", "left_detected", "right_detected", "left_conf", "right_conf", "lat_offset_m"]
    csv_log = CSVWriter(str(csv_path), headers)
    
    # Process First Frame
    processed_frame, debug_images, ll, rl, lat_offset_m = process_frame(
        first_frame, M, Minv, left_line, right_line, xm_per_pix, ym_per_pix, get_debug=True
    )
    
    csv_log.write_frame(ll, rl, lat_offset_m) # <--- Pass real offset
    
    collage_path = output_dir / f"{input_path.stem}_debug_collage.jpg"
    cv2.imwrite(str(collage_path), create_debug_collage(debug_images))
    print(f"Saved debug collage to: {collage_path}")

    # Process Video
    if is_video:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_path = str(output_dir / f"{input_path.stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, img_size)
        
        out.write(processed_frame)
        
        frame_count = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed_frame, _, ll, rl, lat_offset_m = process_frame(
                frame, M, Minv, left_line, right_line, xm_per_pix, ym_per_pix, get_debug=False
            )
            out.write(processed_frame)
            
            csv_log.write_frame(ll, rl, lat_offset_m)
            
            cv2.imshow('Real-time Processing', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if frame_count % 100 == 0: 
                print(f"  ... processed {frame_count} frames")
            frame_count += 1
        
        cap.release()
        out.release()
        print(f"Video processing complete! Saved to {out_path}")
        
    else: # Process Image
        final_path = output_dir / f"{input_path.stem}_annotated.jpg"
        cv2.imwrite(str(final_path), debug_images['final'])
        print(f"Image processing complete! Saved to {final_path}")
    
    csv_log.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lane Detection Pipeline. Run from the project's ROOT directory.")
    parser.add_argument('input', help="Path to the input image or video (e.g., 'data/challenge_video.mp4')")
    parser.add_argument('--output', default='outputs', help="Path to the output directory (e.g., 'outputs')")
    args = parser.parse_args()
    main(args)