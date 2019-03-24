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
from moviepy.editor import VideoFileClip
import sliding_window as sw
import single_image_pipeline as sip

if len(sys.argv) < 3:
    print('usage:')
    print('   python {} clip-name parameter-file-name [length]'.format(sys.argv[0]))
    quit()

l = -1 # default, so we recognize, we want the full length
if len(sys.argv) > 3:
    l = int(sys.argv[3])

c = sip.Configuration()
c.load_config(sys.argv[2])
lstart=0
rstart=0
imgcnt=0
left_fit = None
right_fit = None

def process_image(img):
    global c,lstart,rstart,imgcnt, left_fit, right_fit
    #lstart, rstart, left_fit, right_fit, img = sip.process_image(c,img,image_name=None,show_flag=False,leftstart=lstart,rightstart=rstart,fill_lane=True,save_flag=False)
    lstart, rstart, left_fit, right_fit, img = sip.process_image(c,img,image_name=None,show_flag=False,leftstart=lstart,rightstart=rstart,fill_lane=True,save_flag=False,prev_left_fit=left_fit,prev_right_fit=right_fit)
    #lstart, rstart, left_fit, right_fit, img = sip.process_image(c,img,image_name=None,show_flag=False,leftstart=0,rightstart=0,fill_lane=True,save_flag=False)
    imgcnt += 1
    #print('image #',imgcnt)
    return img

clipname = sys.argv[1]
outclipname = './output_videos/' + clipname

print('CLIP:',clipname,' -> ',outclipname,' |',sys.argv[2])

clip = None
if l > 0:
    clip = VideoFileClip(clipname).subclip(0,l)
else:
    clip = VideoFileClip(clipname)

oclip = clip.fl_image(process_image)
oclip.write_videofile(outclipname, audio=False)
