import time
import numpy as np
import cv2
from fullscreen import FullScreen

def main():
    screen = FullScreen()
    print(screen.shape)

    for i in range(10):
        print(i)
        
        height, width, ch = screen.shape
        image = np.fromfunction(lambda y, x, c: 255*(c==0)+x/width*255*(c==1)+y/height*255*(c==2), (height, width, 3)).astype(np.uint8)
        cv2.putText(image, str(i), (width//7*3, height//7*5), cv2.FONT_HERSHEY_SIMPLEX, width/80, (255, 255, 255), 10, cv2.LINE_AA)

        s = time.time()

        screen.imshow(image)

        f = time.time()
        print(f-s)

if __name__=="__main__":
    main()
