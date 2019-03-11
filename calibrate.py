# reads calibration images; computes calibration data; potentiellay saves the data (as pickle)
#
# this is partially derived from "example.ipynb"
#
import os.path
import sys
import re
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

booleanpattern = re.compile('^\\s*(true|yes|1|on|ja)\\s*$', re.IGNORECASE)
# True if booleanpattern.match(...) else False

# flags/ filenames: set defaults and get command line overwrites
show_flag = False
save_flag = True
pickle_file_name = 'calibration.pickle'

if len(sys.argv) > 4:
    print('usage:')
    print('   python {} [show-flag [save-flag [pickle-file-name]]]'.format(sys.argv[0]))
    quit()

if len(sys.argv) > 1:
    show_flag = True if booleanpattern.match(sys.argv[1]) else False
    if len(sys.argv) > 2:
        save_flag = True if booleanpattern.match(sys.argv[2]) else False
        if len(sys.argv) > 3:
            pickle_file_name = sys.argv[3]

print('showing images:        {}'.format(show_flag))
print('saving images:         {}'.format(save_flag))
print('data- output filename: {}'.format(pickle_file_name))

# we know that the calibration images have 9x6- chess board grids
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# load the names of the calibration images
images = glob.glob('./camera_cal/calibration*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # find corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # only use if we really found the corners
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        if show_flag or save_flag:
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            if save_flag:
                ofname = fname.replace('camera_cal/','output_images/')
                cv2.imwrite(ofname,img)
            if show_flag:
                cv2.imshow('img',img)
                cv2.waitKey(1000)

if show_flag: # or save_flag:
    cv2.destroyAllWindows()

# actual calibration; use one test- image from the respective folder to get the shape of the camera images
testimg = cv2.imread('./test_images/test1.jpg')
imgshape = testimg.shape
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,imageSize=imgshape[::-1][1:3],cameraMatrix=None,distCoeffs=None)#, None, None)

# save for later use
with open(pickle_file_name, 'wb') as f:
    pickledata = ( ret, mtx, dist, rvecs, tvecs )
    pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

    print('calibration data written to "{}"'.format(pickle_file_name))
print('done!')
