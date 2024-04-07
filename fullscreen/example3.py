"""Projector-Camera with simple gray code
"""
import os
import time
import numpy as np
import cv2
import structuredlight as sl # https://github.com/elerac/structuredlight
from fullscreen import FullScreen

def main():
    # Output directory names
    OUTPUT_DIRNAME_BASE = "procam_capture"
    OUTPUT_DIRNAME_CODE = f"{OUTPUT_DIRNAME_BASE}/code"
    OUTPUT_DIRNAME_OBSERVE = f"{OUTPUT_DIRNAME_BASE}/observe"
    os.makedirs(OUTPUT_DIRNAME_CODE, exist_ok=True)
    os.makedirs(OUTPUT_DIRNAME_OBSERVE, exist_ok=True)

    # Camera setup (use your USB camera)
    CAMERA_INDEX = 0
    camera = cv2.VideoCapture(CAMERA_INDEX)

    # Projector setup
    projector = FullScreen()
    height, width = projector.shape[:2]

    # Generate gray code
    gray = sl.Gray()
    imlist_code = gray.generate((width, height))
    
    # Projection and Capture
    imlist_observe = []
    for i, img_code in enumerate(imlist_code):
        # Projection
        projector.imshow(img_code)
        time.sleep(0.2)
        
        # Capture
        ret, img_observe_bgr = camera.read()
        img_observe = cv2.cvtColor(img_observe_bgr, cv2.COLOR_BGR2GRAY)
        imlist_observe.append(img_observe)
        
        # Save images
        cv2.imwrite(f"{OUTPUT_DIRNAME_CODE}/code-{i+1}.png", img_code)
        cv2.imwrite(f"{OUTPUT_DIRNAME_OBSERVE}/observe-{i+1}.png", img_observe)
    
    # Decode gray code
    img_index = gray.decode(imlist_observe, thresh=127)
    
    # Save decoded result
    img_index_u8 = np.clip(img_index / width * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{OUTPUT_DIRNAME_BASE}/index.png", img_index_u8)

if __name__=="__main__":
    main()
