import time
import cv2
from fullscreen import FullScreen

def main():
    screen = FullScreen()

    image = cv2.imread("image/himeji_castle.jpg", 1)

    screen.imshow(image)

    time.sleep(3)

if __name__=="__main__":
    main()
