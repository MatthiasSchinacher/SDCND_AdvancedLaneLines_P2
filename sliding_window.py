# implement sliding window approach (derived from my solution to the quizz)
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def find_lane_pixels(warped,xsplit=0.5,nwindows=15,margin=100,minpix=30,leftstart=0,rightstart=0):
    # check, if this is really binary 1-based
    img = warped
    #print('DEBUG-A img',type(img),img)
    #if np.max(img) > 1:
    #    img = img/255

    # as a strating point, we do a hist. for the bottom third
    h = np.sum(img[2*img.shape[0]//3:,:], axis=0)

    out_img = None
    if len(img.shape)<3:
        out_img = np.dstack((img,img,img))
    elif img.shape[2] == 1:
        out_img = np.dstack((img,img,img))
    else:
        out_img = np.copy(img)
        h = np.sum(h, axis=1) # in this case, did not have a single color channel :-)

    #print('DEBUG-X h',type(h),h.shape,h)
    m = int(xsplit * float(h.shape[0]))

    left0 = leftstart
    tmp = np.argmax(h[:m])
    if tmp > 0:
        if left0 > 0:
            left0 = int(0.3*float(left0) + 0.7*float(tmp))
        else:
            left0 = tmp
    if left0 == 0:
        left0 = m/2

    right0 = rightstart
    tmp = np.argmax(h[m:])
    if tmp > 0:
        if right0 > 0:
            right0 = int(0.3*float(right0) + 0.7*float(m+tmp))
        else:
            right0 = m+tmp
    if right0 == 0:
        right0 = m + m/3

    #print('DEBUG-0 h',type(h),h.shape,h,'|',m,left0,right0)
    #quit()

    wh = np.int(img.shape[0]//nwindows)

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left = left0
    right = right0

    # lists left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # looping through the windows
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*wh
        win_y_high = img.shape[0] - window*wh
        win_xleft_low = left - margin
        win_xleft_high = left + margin
        win_xright_low = right - margin
        win_xright_high = right + margin

        # Draw the windows on the visualization image
        #print('DEBUG: ',type(out_img.shape),out_img.shape)
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2)

        good_left_inds = []
        good_right_inds = []
        for i in range(len(nonzeroy)):
            if nonzeroy[i] >= win_y_low and nonzeroy[i] <= win_y_high:
                if nonzerox[i] >= win_xleft_low and nonzerox[i] <= win_xleft_high:
                    good_left_inds.append(i)
                if nonzerox[i] >= win_xright_low and nonzerox[i] <= win_xright_high:
                    good_right_inds.append(i)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            #print(window,' recenter left with ',len(good_left_inds),' => ',minpix)
            tmp = [nonzerox[i] for i in good_left_inds]
            left = int(float(sum(tmp))/float(len(tmp)))
        if len(good_right_inds) > minpix:
            #print(window,' recenter right with ',len(good_right_inds),' => ',minpix)
            tmp = [nonzerox[i] for i in good_right_inds]
            right = int(float(sum(tmp))/float(len(tmp)))

    try:
        #print('DEBUG-1 left_lane_inds:',type(left_lane_inds),left_lane_inds)
        left_lane_inds = np.concatenate(left_lane_inds)
        left_lane_inds = np.array(left_lane_inds,dtype=np.int32)
        right_lane_inds = np.concatenate(right_lane_inds)
        right_lane_inds = np.array(right_lane_inds,dtype=np.int32)
    except ValueError:
        pass

    # Extract left and right line pixel positions
    #print('DEBUG-2 left_lane_inds:',type(left_lane_inds),left_lane_inds)
    leftx = []
    lefty = []
    rightx = []
    righty = []
    if left_lane_inds.shape[0] > 0:
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
    if right_lane_inds.shape[0] > 0:
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img, left0, right0
