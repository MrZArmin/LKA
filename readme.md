# Lane Keeping Assist (LKA) - Lane Detection Pipeline

A computer vision-based lane detection system for autonomous driving applications. This project processes video footage or images to detect lane lines, calculate vehicle position, and compute road curvature metrics.

## 🎯 Features

- **Real-time Lane Detection**: Processes video frames to identify left and right lane boundaries
- **Perspective Transform**: Warps road images to bird's-eye view for accurate lane fitting
- **Adaptive Detection**: Uses temporal smoothing to maintain stable lane tracking
- **Metrics Calculation**: 
  - Lateral offset from lane center (meters)
  - Road curvature radius (meters)
  - Lane detection confidence scores
- **Debug Visualization**: Generates debug collages showing pipeline stages
- **CSV Logging**: Exports per-frame metrics for analysis

## 📋 Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy

## 🚀 Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/LKA
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy
   ```

3. **Verify project structure**:
   ```
   LKA/
   ├── run_pipeline.py          # Main entry point
   ├── readme.md                # This file
   ├── data/                    # Input videos/images
   │   ├── challenge_video.mp4
   │   └── road.png
   ├── outputs/                 # Generated outputs
   └── src/                     # Source modules
       ├── preprocess.py        # Image preprocessing
       ├── warp.py              # Perspective transformation
       ├── lane_fit.py          # Lane polynomial fitting
       ├── overlay.py           # Final visualization
       ├── temporal.py          # Temporal smoothing
       ├── metrics.py           # Curvature & offset calculations
       ├── csv_writer.py        # CSV logging
       └── debug_utils.py       # Debug visualizations
   ```

## 📖 Usage

### Basic Usage

Run the pipeline on a video:
```bash
python run_pipeline.py data/challenge_video.mp4
```

Run the pipeline on an image:
```bash
python run_pipeline.py data/road.png
```

### Custom Output Directory

Specify a custom output folder:
```bash
python run_pipeline.py data/challenge_video.mp4 --output my_results
```

### Command Line Arguments

```bash
python run_pipeline.py <input_path> [--output OUTPUT_DIR]
```

- `input_path`: Path to input video (`.mp4`, `.avi`, `.mov`) or image (`.jpg`, `.png`)
- `--output`: Output directory (default: `outputs/`)

## 🎮 Interactive Calibration

When you run the pipeline, you'll be prompted to calibrate the perspective transform:

1. **A window will appear** showing the first frame
2. **Click 4 points** to define the lane trapezoid:
   - Top-left corner of the lane
   - Top-right corner of the lane
   - Bottom-right corner of the lane
   - Bottom-left corner of the lane
3. **Press any key** to confirm and start processing

**Tip**: Select points that form a trapezoid around a straight section of the lane for best results.

## 📊 Output Files

After processing, the following files are generated in the `outputs/` directory:

1. **`<filename>_annotated.mp4`** (for videos): Processed video with lane overlay, curvature, and offset displayed
2. **`<filename>_annotated.jpg`** (for images): Processed image with lane overlay
3. **`<filename>_debug_collage.jpg`**: Debug visualization showing:
   - Original frame
   - Binary mask (preprocessing)
   - Warped bird's-eye view
   - Final annotated result
4. **`<filename>_per_frame.csv`**: Per-frame metrics log with columns:
   - `frame_id`: Frame number
   - `left_detected`: Left lane detected (True/False)
   - `right_detected`: Right lane detected (True/False)
   - `left_conf`: Left lane confidence score
   - `right_conf`: Right lane confidence score
   - `lat_offset_m`: Lateral offset from lane center (meters)

## 🏫 Academic Context

**Course**: Járművek és szenzorok (Vehicles and Sensors)  
**Institution**: ELTE (Eötvös Loránd University)  
**Project**: Lane Keeping Assist System

## 📄 License

This project is for educational purposes.


**Last Updated**: November 2025
