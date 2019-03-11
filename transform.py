# computes perspective transformation and saves as pickle file; potentially applies to undistorted files and shows/ saves them
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
save_flag = False

if len(sys.argv) > 5 or len(sys.argv) < 3:
    print('usage:')
    print('   python {} transformation-name coordinates [images-show-flag [images-save-flag]]'.format(sys.argv[0]))
    quit()

tname = sys.argv[1]
pickle_file_name = tname + '.pickle'
coordinates = sys.argv[2]

if len(sys.argv) > 3:
    show_flag = True if booleanpattern.match(sys.argv[3]) else False
    if len(sys.argv) > 4:
        save_flag = True if booleanpattern.match(sys.argv[4]) else False

print('transformation-name: {} (file: {})'.format(tname,pickle_file_name))
print('coordinates:         {}'.format(coordinates))
print('showing images:      {}'.format(show_flag))
print('saving images:       {}'.format(save_flag))

tmpc = coordinates.split()
if len(tmpc) != 16:
    print('the coordinates string must be 16 numbers!')
    quit()

x0   = int(tmpc[0])
y0   = int(tmpc[1])
x1   = int(tmpc[2])
y1   = int(tmpc[3])
x2   = int(tmpc[4])
y2   = int(tmpc[5])
x3   = int(tmpc[6])
y3   = int(tmpc[7])
x4   = int(tmpc[8])
y4   = int(tmpc[9])
x5   = int(tmpc[10])
y5   = int(tmpc[11])
x6   = int(tmpc[12])
y6   = int(tmpc[13])
x7   = int(tmpc[14])
y7   = int(tmpc[15])

src  = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
dst  = np.float32([[x4,y4],[x5,y5],[x6,y6],[x7,y7]])
M    = cv2.getPerspectiveTransform(src, dst)
Mrev = cv2.getPerspectiveTransform(dst, src)

with open(pickle_file_name, 'wb') as f:
    pickledata = ( M, Mrev )
    pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

    print('transformation data written to "{}"'.format(pickle_file_name))

if show_flag or save_flag:
    # load the names of the test images
    images = glob.glob('./output_images/*undistorted.jpg')
    for fname in images:
        img  = cv2.imread(fname)
         # copy for image with rectangle
        imgr  = cv2.imread(fname)
        color=[255,0,0]
        thickness=2
        cv2.line(imgr, (x0, y0), (x1, y1), color, thickness)
        cv2.line(imgr, (x1, y1), (x2, y2), color, thickness)
        cv2.line(imgr, (x2, y2), (x3, y3), color, thickness)
        cv2.line(imgr, (x3, y3), (x0, y0), color, thickness)

        #the actual warped/ transformed image
        imgw = cv2.warpPerspective(img, M, img.shape[::-1][1:3],flags=cv2.INTER_LINEAR) # TODO: is there other shapes?

        if save_flag:
            ofname = fname.replace('_undistorted.jpg','_undistorted_' + tname + '_R.jpg')
            cv2.imwrite(ofname,imgr)
            ofname = fname.replace('_undistorted.jpg','_undistorted_warped_' + tname + '.jpg')
            cv2.imwrite(ofname,imgw)

        if show_flag:
            cv2.imshow('img',img)
            cv2.waitKey(1000)
            cv2.imshow('img',imgr)
            cv2.waitKey(1000)
            cv2.imshow('img',imgw)
            cv2.waitKey(1000)

        if save_flag or show_flag:
            color=[0,0,255]
            thickness=2
            cv2.line(imgw, (x4, y4), (x5, y5), color, thickness)
            cv2.line(imgw, (x5, y5), (x6, y6), color, thickness)
            cv2.line(imgw, (x6, y6), (x7, y7), color, thickness)
            cv2.line(imgw, (x7, y7), (x4, y4), color, thickness)

            if save_flag:
                ofname = fname.replace('_undistorted.jpg','_undistorted_warped_' + tname + '_R.jpg')
                cv2.imwrite(ofname,imgw)

            if show_flag:
                cv2.imshow('img',imgw)
                cv2.waitKey(1000)

if show_flag:
    cv2.destroyAllWindows()

print('done!')
