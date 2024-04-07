import cv2
from PIL import Image

cam_id=1
fps=0.5
exp_val = -6
capture_width=4024
capture_height=3036
show_ratio = 0.3

show_width = int(capture_width*show_ratio)
show_height = int(capture_height*show_ratio)

print("Initializing camera")
camera = cv2.VideoCapture(cam_id)
codec = 0x47504A4D # MJPG
print("Setting camera mode")
camera.set(cv2.CAP_PROP_FPS, fps)
camera.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
camera.set(cv2.CAP_PROP_EXPOSURE, exp_val)

print("Start capture")
while(True):
    ret,frame = camera.read()
    frame = cv2.resize(frame,(show_width,show_height))
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(frame.shape)
        break
    
camera.release()
cv2.destroyAllWindows()