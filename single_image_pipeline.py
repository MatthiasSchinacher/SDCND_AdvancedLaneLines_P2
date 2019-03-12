# implements the lane finding pipeline on single images
#
import os.path
import sys
import re
import configparser
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sliding_window as sw

booleanpattern = re.compile('^\\s*(true|yes|1|on|ja)\\s*$', re.IGNORECASE)
# True if booleanpattern.match(...) else False

class Configuration():
    def __init__(self):
        # flags, filenames, parameters: set defaults and get command line overwrites
        self.calfilename = 'calibration.pickle'
        self.tname     = 'M6'
        self.thresh_S_low  = 90      # lower S- channel threshold
        self.thresh_S_high = 255     # upper S- channel threshold
        self.blur_kernel = 5
        self.sobel_kernel_m=5        # sobel kernel for magnitude
        self.sobel_thresh_m_low=50
        self.sobel_thresh_m_high=155
        self.sobel_kernel_d=7        # sobel kernel for direction
        self.sobel_thresh_d_low=0.8
        self.sobel_thresh_d_high=1.2
        self.canny_low  = 30
        self.canny_high = 100
        # hough lines ...
        self.rho = 1
        self.theta = np.pi/180.0
        self.threshold = 3
        self.min_line_len = 20
        self.max_line_gap = 5

        self.ret   = None
        self.mtx   = None
        self.dist  = None
        self.rvecs = None
        self.tvecs = None

        self.M = None
        self.Mrev = None

    def load_config(self,filename):
        print('loading config ...')
        config = configparser.ConfigParser()
        config.read(filename)

        if 'params' in config:
            params = config['params']
            if 'tname' in params:
                self.tname = params['tname']
            if 'calfilename' in params:
                self.calfilename = params['calfilename']
            if 'thresh_S_low' in params:
                self.thresh_S_low = int(params['thresh_S_low'])
            if 'thresh_S_high' in params:
                self.thresh_S_high = int(params['thresh_S_high'])
            if 'blur_kernel' in params:
                self.blur_kernel = int(params['blur_kernel'])
            if 'sobel_kernel_m' in params:
                self.sobel_kernel_m = int(params['sobel_kernel_m'])
            if 'sobel_thresh_m_low' in params:
                self.sobel_thresh_m_low = int(params['sobel_thresh_m_low'])
            if 'sobel_thresh_m_high' in params:
                self.sobel_thresh_m_high = int(params['sobel_thresh_m_high'])
            if 'sobel_kernel_d' in params:
                self.sobel_kernel_d = int(params['sobel_kernel_d'])
            if 'sobel_thresh_d_low' in params:
                self.sobel_thresh_d_low = float(params['sobel_thresh_d_low'])
            if 'sobel_thresh_d_high' in params:
                self.sobel_thresh_d_high = float(params['sobel_thresh_d_high'])
            if 'canny_low' in params:
                self.canny_low = int(params['canny_low'])
            if 'canny_high' in params:
                self.canny_high = int(params['canny_high'])

        if 'hough_params' in config:
            hough_params = config['hough_params']
            if 'rho' in hough_params:
                self.rho = int(hough_params['rho'])
            if 'theta' in hough_params:
                self.theta = float(hough_params['theta'])
            if 'threshold' in hough_params:
                self.threshold = int(hough_params['threshold'])
            if 'min_line_len' in hough_params:
                self.min_line_len = int(hough_params['min_line_len'])
            if 'max_line_gap' in hough_params:
                self.max_line_gap = int(hough_params['max_line_gap'])

        with open(self.calfilename, 'rb') as f:
            tmp = pickle.load(f)
            ( self.ret, self.mtx, self.dist, self.rvecs, self.tvecs ) = tmp

        with open(self.tname + '.pickle', 'rb') as f:
            tmp = pickle.load(f)
            (self.M,self.Mrev) = tmp

        #print('mtx, dist, blur_kernel:',self.mtx,self.dist,self.blur_kernel)
        #print('loading config done!')


# the actual pipeline
def process_image(c,image,image_name=None,show_flag=False,leftstart=0,rightstart=0,fill_lane=False,save_flag=False,prev_left_fit=None,prev_right_fit=None,save_all=False):
    #print('... processing imgage')
    original_img = image
    #print('mtx, dist, blur_kernel:',c.mtx,c.dist,c.blur_kernel)
    if show_flag:
        cv2.imshow('img',image)
        cv2.waitKey(500)
    if save_all:
        ofname = image_name.replace('.jpg','-01.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,image)

# undistort
    img = cv2.undistort(image, c.mtx, c.dist, None, c.mtx)
    if show_flag:
        #print('undistorted image; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    if save_all:
        ofname = image_name.replace('.jpg','-U.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

## greyscale
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    cv2.imshow('img',img)
#    cv2.waitKey(1500)

# S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    if show_flag:
        print('S- channel image; shape: ',type(S.shape),S.shape)
        cv2.imshow('img',S)
        cv2.waitKey(1000)
    if save_all:
        ofname = image_name.replace('.jpg','-S.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,S)

# thresholding the S channel
    if c.thresh_S_high > 0:
        tmpimg = np.zeros_like(S)
        tmpimg[(S > c.thresh_S_low) & (S <= c.thresh_S_high)] = 1
        img = np.uint8(255*tmpimg/np.max(tmpimg))
        if show_flag:
            print('thresholded S- channel image; shape: ',type(img.shape),img.shape)
            cv2.imshow('img',img)
            cv2.waitKey(1000)
        if save_all:
            ofname = image_name.replace('.jpg','-S2.jpg').replace('./test_images/','')
            cv2.imwrite(ofname,img)
    else:
        img = S

# blur
    if c.blur_kernel > 0:
        img = cv2.GaussianBlur(img, (c.blur_kernel, c.blur_kernel), 0)
        if show_flag:
            print('blurred image; shape: ',type(img.shape),img.shape)
            cv2.imshow('img',img)
            cv2.waitKey(500)
        if save_all:
            ofname = image_name.replace('.jpg','-B.jpg').replace('./test_images/','')
            cv2.imwrite(ofname,img)

# sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=c.sobel_kernel_m)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=c.sobel_kernel_m)
    sobel_m = np.sqrt(np.add(np.multiply(sobelx,sobelx),np.multiply(sobely,sobely)))
    sobel_m = np.uint8(255*sobel_m/np.max(sobel_m)) # magnitude
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=c.sobel_kernel_d)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=c.sobel_kernel_d)
    sobel_d = np.arctan2(sobely, sobelx) # direction
    tmpimg = np.zeros_like(sobel_m)
    tmpimg[((sobel_m >= c.sobel_thresh_m_low) & (sobel_m <= c.sobel_thresh_m_high)) | ((sobel_d >= c.sobel_thresh_d_low) & (sobel_d <= c.sobel_thresh_d_high))] = 1
    img = np.uint8(255*tmpimg/np.max(tmpimg))
    if show_flag:
        print('thresholded sobel image; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(1000)
    if save_all:
        ofname = image_name.replace('.jpg','-SOB.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

# canny
    if c.canny_high > 0:
        img = cv2.Canny(img, c.canny_low, c.canny_high)
        if show_flag:
            print('canny image; shape: ',type(img.shape),img.shape)
            cv2.imshow('img',img)
            cv2.waitKey(1000)
        if save_all:
            ofname = image_name.replace('.jpg','-C.jpg').replace('./test_images/','')
            cv2.imwrite(ofname,img)
            #return

# mask
    xmin = 0
    xmax = img.shape[1]
    ymin = 0
    ymax = img.shape[0]

    xstart = int(float(xmin) + 2.0 * float(xmax - xmin) / 17.0)
    x0 = int(float(xmin) + 4.0 * float(xmax - xmin) / 16.0)
    x1 = int(float(xmin) + 18.0 * float(xmax - xmin) / 39.0)
    x2 = int(float(xmin) + 21.0 * float(xmax - xmin) / 39.0)
    x3 = int(float(xmin) + 12.0 * float(xmax - xmin) / 16.0)
    xend = int(float(xmin) + 15.0 * float(xmax - xmin) / 17.0)
    y0 = int(float(ymin) + 16.0 * float(ymax - ymin) / 20.0)
    y1 = int(float(ymin) + 19.0 * float(ymax - ymin) / 40.0)
    ystart = int(float(ymax) - 5.0 * float(ymax - ymin) / 37.0)
    verts = np.array([[(xstart,ystart),(x0,y0),(x1, y1),(x2, y1),(x3,y0),(xend,ystart)]], dtype=np.int32)
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, verts, ignore_mask_color)
    img = cv2.bitwise_and(img,mask)

    if show_flag:
        print('masked image; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    if save_all:
        ofname = image_name.replace('.jpg','-M.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

# hough lines
    lines = cv2.HoughLinesP(img, c.rho, c.theta, c.threshold, np.array([]), minLineLength=c.min_line_len, maxLineGap=c.max_line_gap)
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color=[255,255,255]
    thickness=2
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    if show_flag:
        print('hough lines image; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    if save_all:
        ofname = image_name.replace('.jpg','-H.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

# transformed
    img = cv2.warpPerspective(img, c.M, img.shape[::-1][1:3]) # only if we have 3 channels
    if show_flag:
        print('warped image; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(500)
    if save_all:
        ofname = image_name.replace('.jpg','-W.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

# sliding window search
    lstart = leftstart
    rstart = rightstart
    leftx, lefty, rightx, righty, img, lstart, rstart = sw.find_lane_pixels(img,xsplit=0.4,nwindows=15,margin=80,minpix=10,leftstart=lstart,rightstart=rstart)

# we initialize our fits with the fits from last time
    left_fit = prev_left_fit
    right_fit = prev_right_fit
    if len(leftx) > 7 and len(lefty) > 7:
        if left_fit is None:
            left_fit = np.polyfit(lefty,leftx,2)
        else:
            left_fit = (0.3* left_fit) + (0.7* np.polyfit(lefty,leftx,2))
    elif left_fit is None: # special case, where we do not have a fit from prevoius AND we can't compute one now
        left_fit = np.array([0.0,0.0,0.2*float(img.shape[1])])

    if len(rightx) > 7 and len(righty) > 7:
        if right_fit is None:
            right_fit = np.polyfit(righty,rightx,2)
        else:
            right_fit = (0.3* right_fit) + (0.7* np.polyfit(righty,rightx,2))
    elif right_fit is None: # special case, where we do not have a fit from prevoius AND we can't compute one now
        right_fit = np.array([0.0,0.0,0.7*float(img.shape[1])])

# "calculating" the curvature:
    y_eval = float(img.shape[0] -1)
    Al = float(left_fit[0])
    Bl = float(left_fit[1])
    Cl = float(left_fit[2])
    left_curverad = "-inf"
    if abs(Al) > 0.0:
        left_curverad = np.sqrt(0.25*((1.0 + (2.0*Al*float(y_eval) + Bl)**2)**3)/Al**2)
    Ar = float(right_fit[0])
    Br = float(right_fit[1])
    Cr = float(right_fit[2])
    right_curverad = "-inf"
    if abs(Ar) > 0.0:
        right_curverad = np.sqrt(0.25*((1.0 + (2.0*Ar*float(y_eval) + Br)**2)**3)/Ar**2)

    if show_flag:
        print('left_curverad:',left_curverad,'; right_curverad:',right_curverad)
        print('sliding windows; shape: ',type(img.shape),img.shape)
        cv2.imshow('img',img)
        cv2.waitKey(1000)
    if save_all:
        ofname = image_name.replace('.jpg','-SW.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

    if show_flag or save_flag:# or True:
        revimg = cv2.warpPerspective(img, c.Mrev, img.shape[::-1][1:3]) # only if we have 3 channels
        revimg = cv2.addWeighted(original_img,0.7,revimg,3.0,0.1)
        if show_flag:
            print('unwarped sliding windows; shape: ',type(revimg.shape),revimg.shape)
            cv2.imshow('img',revimg)
            cv2.waitKey(1000)
        if save_flag:
            ofname = image_name.replace('.jpg','_boxes.jpg').replace('/test_images/','/output_images/')
            cv2.imwrite(ofname,revimg)

    if fill_lane:
        # we want to fill the area between the left and right lane fits
        ximg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        tmpverts = []

        ymax = int(0.9*ximg.shape[1])
        ymin = int(0.15*ximg.shape[1])
        y = ymax
        x = int(Al*float(y**2)+Bl*float(y)+Cl)
        tmpverts.append( (x,y) )
        while y>=ymin:
            x = int(Ar*float(y**2)+Br*float(y)+Cr)
            tmpverts.append( (x,y) )
            y = y -10
        y = y+10
        while y<ymax:
            x = int(Al*float(y**2)+Bl*float(y)+Cl)
            tmpverts.append( (x,y) )
            y = y +10

        xverts = np.array([tmpverts], dtype=np.int32)
        xcolor=[0,255,0]
        cv2.fillPoly(ximg, xverts, xcolor)
        ximg = cv2.warpPerspective(ximg, c.Mrev, img.shape[::-1][1:3])
        img = cv2.addWeighted(original_img,0.9,ximg,0.3,0.1)
        #img = cv2.addWeighted(img,0.9,revimg,0.3,0.1)
        if show_flag:
            print('identified lane?; shape: ',type(ximg.shape),ximg.shape)
            cv2.imshow('img',img)
            cv2.waitKey(1000)
        if save_flag:
            ofname = image_name.replace('.jpg','_L.jpg').replace('/test_images/','/output_images/')
            cv2.imwrite(ofname,img)

    if save_all:
        ofname = image_name.replace('.jpg','-final.jpg').replace('./test_images/','')
        cv2.imwrite(ofname,img)

    return lstart, rstart, left_fit, right_fit, img

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage:')
        print('   python {} parameter-file-name [test-image]'.format(sys.argv[0]))
        quit()

    c = Configuration()
    c.load_config(sys.argv[1])

    if len(sys.argv) > 2:
        fname = './test_images/' + sys.argv[2]
        img = cv2.imread(fname)
        print('original image ("' + fname + '"); shape: ',type(img.shape),img.shape)
        process_image(c,img,image_name=fname,show_flag=True,fill_lane=True,save_flag=True,save_all=True)
    else:
        images = glob.glob('./test_images/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            print('original image ("' + fname + '"); shape: ',type(img.shape),img.shape)
            process_image(c,img,image_name=fname,show_flag=True,fill_lane=True,save_flag=True)

    cv2.waitKey(5000)
    cv2.destroyAllWindows()
