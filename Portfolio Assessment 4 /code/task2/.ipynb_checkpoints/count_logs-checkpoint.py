import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def count_logs(image_path, model_path="log_segmentation_model.h5"):
    model = load_model(model_path)
    
    img = cv2.imread(image_path)
    original_size = img.shape[:2]
    
    resized_img = cv2.resize(img, (256, 256))
    resized_img = resized_img / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)
    
    pred_mask = model.predict(resized_img)[0]
    
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]))
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Use: python count_logs.py <image_directory>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    num_logs = count_logs(image_path)
    print(f"The number of logs in image {os.path.basename(image_path)}: {num_logs}")