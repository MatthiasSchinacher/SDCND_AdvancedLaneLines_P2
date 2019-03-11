# reads calibration data/ parameters and computes undistorted images (potentially saved)
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

print('showing images:      {}'.format(show_flag))
print('saving images:       {}'.format(save_flag))
print('paramdata- filename: {}'.format(pickle_file_name))

if show_flag or save_flag:
    with open(pickle_file_name, 'rb') as f:
        print('parameter- data file successfully opened ...')
        pickledata = pickle.load(f)
        ( ret, mtx, dist, rvecs, tvecs ) = pickledata

        # load the names of the test images
        images = glob.glob('./test_images/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            imgu = cv2.undistort(img, mtx, dist, None, mtx)

            if save_flag:
                ofname = fname.replace('test_images/','output_images/').replace('.jpg','_undistorted.jpg')
                cv2.imwrite(ofname,imgu)
            if show_flag:
                cv2.imshow('img',img)
                cv2.waitKey(1000)
                cv2.imshow('img',imgu)
                cv2.waitKey(1000)
else:
    print('no showing or saving images? => nothing to do')

if show_flag:
    cv2.destroyAllWindows()

print('done!')
