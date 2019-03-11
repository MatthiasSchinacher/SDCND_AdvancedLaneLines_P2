# read calibration parameters and write undistorted calibration pictures
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

pickle_file_name = 'calibration.pickle'
if len(sys.argv) > 1:
    pickle_file_name = sys.argv[1]

with open(pickle_file_name, 'rb') as f:
    pickledata = pickle.load(f)
    ( ret, mtx, dist, rvecs, tvecs ) = pickledata

    images = glob.glob('./camera_cal/calibration*.jpg')
    for fname in images:
        img  = cv2.imread(fname)
        imgu = cv2.undistort(img, mtx, dist, None, mtx)
        ofname = fname.replace('camera_cal/','output_images/').replace('.jpg','_undistorted.jpg')
        cv2.imwrite(ofname,imgu)

    print('done!')
