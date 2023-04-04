#!/bin/bash

# Manual setting for USBCam UCAM-C820ABBK
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 -c focus_absolute=100

v4l2-ctl -d /dev/video0 -c auto_exposure=1
v4l2-ctl -d /dev/video0 -c exposure_time_absolute=75
  
v4l2-ctl -d /dev/video0 -c white_balance_automatic=0
v4l2-ctl -d /dev/video0 -c white_balance_temperature=4600

v4l2-ctl -d /dev/video0 -L
