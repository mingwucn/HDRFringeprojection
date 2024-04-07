import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()

canvas = np.zeros((480, 640))
screen = pf.screen(canvas, 'Sinusoid')

start = time.time()
n = 34
while n < 120:
    n = n + 1
    now = time.time() - start

    x = np.linspace(now-2, now, 100)
    y = np.sin(2*np.pi*x) + np.sin(3*np.pi*x)
    plt.xlim(now-2, now+1)
    plt.ylim(-3, 3)
    plt.plot(x, y, c='black')

    # If we haven't already shown or saved the plot, then we need to draw the figure first...
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    screen.update(image)

# screen.close()
input("Press Enter to close...")