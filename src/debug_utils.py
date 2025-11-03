# src/debug_utils.py
import numpy as np
import cv2

def create_debug_collage(image_dict):
    """
    Creates a 2x2 collage from a dictionary of 4 images.
    Expects dict keys: 'original', 'binary', 'warped', 'final'
    """
    try:
        # Ensure all images are 3-channel BGR for stacking
        orig = image_dict['original']
        final = image_dict['final']
        
        # Convert 1-channel binary to 3-channel
        binary_mask = image_dict['binary']
        if len(binary_mask.shape) == 2:
            binary = np.dstack((binary_mask, binary_mask, binary_mask)) * 255
        else:
            binary = binary_mask

        # Warped image is already 3-channel
        warped = image_dict['warped']

        # Resize all images to a consistent size for the collage
        # Using a 16:9 aspect ratio
        h, w = 360, 640 
        
        img_list = []
        titles = ['1. Original', '2. Binary Mask', '3. Warped Fit', '4. Final Overlay']
        
        for i, img in enumerate([orig, binary, warped, final]):
            # Resize
            img_resized = cv2.resize(img, (w, h))
            # Add text title to each image
            cv2.putText(img_resized, titles[i], (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            img_list.append(img_resized)

        # Assemble the 2x2 grid
        top_row = np.hstack((img_list[0], img_list[1]))
        bottom_row = np.hstack((img_list[2], img_list[3]))
        collage = np.vstack((top_row, bottom_row))
        
        return collage
        
    except Exception as e:
        print(f"Error creating collage: {e}")
        # Return a black image on failure
        return np.zeros((720, 1280, 3), dtype=np.uint8)