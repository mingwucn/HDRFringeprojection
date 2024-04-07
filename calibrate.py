
import argparse
import os

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-x', '--width', type=int, default=18)
parser.add_argument('-y', '--height', type=int, default=9)
parser.add_argument('-s', '--size', type=float, default=1.0)
parser.add_argument('-t', '--temp')
parser.add_argument('image', nargs='+')

args = parser.parse_args()

pattern_size = (args.width, args.height)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32 )
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= args.size


obj_points = []
img_points = []


for image_i, path in enumerate(args.image):

    # print path
    image = cv2.imread(path)
    w, h = image.shape[:2]
    image = cv2.resize(image, (1024, int(1024 * w / h)))

    # print '  finding rough corners'
    found, corners = cv2.findChessboardCorners(image, pattern_size, 0, cv2.CV_CALIB_CB_ADAPTIVE_THRESH)
    if found:
        # print '  found', corners.shape

        # print 'finding fine corners'
        # TODO: move this back to the large frame.
        # print '  refine corners'
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # cv2.cornerSubPix(image[...,1], corners, (5, 5), (-1,-1), criteria)

        cv2.drawChessboardCorners(image, pattern_size, corners, found)

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

    if args.temp:
        new_path = os.path.join(args.temp, os.path.basename(path))
        cv2.imwrite(new_path, image)

# print '---'

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[:2], None, None)
# print "RMS:", rms
# print "camera matrix:\n", camera_matrix
# print "distortion coefficients: ", dist_coefs.ravel()
# print "rvecs: ", rvecs
# print "tvecs: ", tvecs

# print '---'

fovx, fovy, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(camera_matrix, image.shape[:2], 23.9, 35.8)
# print 'fovx:', fovx
# print 'fovy:', fovy
# print 'focal_length:', focal_length
# print 'principal_point:', principal_point
# print 'aspect_ratio:', aspect_ratio


if args.temp:
    for image_i, path in enumerate(args.image):
        image = cv2.imread(path)
        w, h = image.shape[:2]
        image = cv2.resize(image, (1024, int(1024 * w / h)))
        image = cv2.undistort(image, camera_matrix, dist_coefs)

        new_path = os.path.join(args.temp, os.path.basename(path) + '-square.jpg')
        cv2.imwrite(new_path, image)