#!/usr/bin/env python3
"""
This script prints the simulated example structures
use `fp23d example_plane.png` or any of the other structures to see how good it works
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import fp23dpy
from fp23dpy.examples import example_plane, example_cylinder, example_cone, example_drop


def example_structure_print(module):
    name = "example_{}".format(module.name)
    print("Printing {}".format(name))

    projected_coordinate_grid = module.get_projected_coordinate_grid()
    calibration = module.get_calibration()

    # First, plot third dimension values
    plt.figure()
    plt.imshow(projected_coordinate_grid[2])
    plt.colorbar(label="z")
    plt.savefig("{}_shape.png".format(name))
    plt.close()

    # Second, print the correct 3D structure with a texture corresponding to a fringe projection
    carrier = fp23dpy.make_carrier(module.shape, calibration["T"], calibration["gamma"])
    texture = np.round(255.0 / 2 * (np.cos(carrier) + 1)).astype(np.uint8)
    fp23dpy.export3D(
        "{}.glb".format(name),
        {"name": name, "grid": projected_coordinate_grid, "texture": texture},
    )

    # Third, print a simulated fringe pattern with segmentation
    signal = 255 / 2 * module.render()
    if np.ma.isMaskedArray(signal) and not isinstance(signal.mask, np.bool_):
        signal.data[signal.mask] = 0
        io.imsave("{}.png".format(name), signal.data.astype(np.uint8))
        segmentation = ~signal.mask
        io.imsave(
            "segmented_{}.png".format(name),
            segmentation.astype(np.uint8) * 255,
            check_contrast=False,
        )
    else:
        io.imsave("{}.png".format(name), signal.astype(np.uint8))

    calibration.write("calibration_{}.txt".format(name))


if __name__ == "__main__":
    modules = [example_plane, example_cylinder, example_cone, example_drop]
    for module in modules:
        example_structure_print(module)
