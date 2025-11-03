# src/csv_writer.py
import csv

class CSVWriter:
    def __init__(self, filepath, headers):
        """
        Initializes the CSV writer.
        filepath: path to the output .csv file
        headers: a list of strings for the header row
        """
        self.filepath = filepath
        self.headers = headers
        
        try:
            # Open the file in 'write' mode ('w')
            self.file = open(self.filepath, 'w', newline='')
            self.writer = csv.writer(self.file)
            
            # Write the header row
            self.writer.writerow(self.headers)
            self.frame_id = 0
            
        except IOError as e:
            print(f"Error opening CSV file: {e}")
            self.file = None
            self.writer = None

    def write_frame(self, left_line, right_line, lat_offset_m=0.0):
        """
        Writes a new row of data for the current frame.
        """
        if self.writer is None:
            return # CSV failed to open
            
        # Get data from the Line objects
        left_detected = 1 if left_line.detected else 0
        right_detected = 1 if right_line.detected else 0
        left_conf = left_line.confidence
        right_conf = right_line.confidence
        
        # Prepare the row
        row = [
            self.frame_id,
            left_detected,
            right_detected,
            f"{left_conf:.2f}",
            f"{right_conf:.2f}",
            f"{lat_offset_m:.2f}"
        ]
        
        # Write the row and increment frame counter
        self.writer.writerow(row)
        self.frame_id += 1

    def close(self):
        """
        Closes the CSV file.
        """
        if self.file is not None:
            self.file.close()
            print(f"CSV data saved to: {self.filepath}")