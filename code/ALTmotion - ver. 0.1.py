#!python3

# A video-based motion detection implementation in Python.
# Works with Raspberry Pi (https://www.raspberrypi.org/)
# and Pi Camera (e.g. https://www.raspberrypi.org/products/pi-noir-camera-v2/).
# The motion detection code is fast, it takes less than 1 ms for it to complete
# for 1280x720 video frame size on Raspbery Pi 3B,
# which allows detecting movements having short duration.

# Note that both the 'sSAD' and the 'motionEstimate' measures of motion (see below)
# can capture motion in a scene observed by a camera,
# including movements associated with respiration and heartbeats.
# See https://github.com/lvetech/ALT for description (with code examples)
# of two implementations of the "Artificial Light Texture (ALT)" technology for
# non-contact real-time detection of heartbeats and respiration.
# These implementations are based on the 'sSAD' and similar measures of motion.
# ALT can use the 'motionEstimate' measure of motion
# for the purpose of detecting respiration and/or heartbeats too.

# Note that the 'sSAD' and the 'motionEstimate' measures of motion
# can exhibit different sensitivity to the same motion
# captured in a sequence of video frames.

# Note that the "baseline" of the 'sSAD' values can be in the range
# of hundreds of thousands while
# the 'sSAD' signal component related to the movements in the scene can have
# a small (e.g. just several percent) amplitude relative to the "baseline".

# Note that the 'camera.framerate' parameter (see below)
# is generally inversely proportional to the minimum duration
# of a movement for which the movement can be (reliably)
# captured in a sequence of video frames and detected by the code below.
# This relation means that, for example,
# 'camera.framerate = 49' setting will allow you to capture
# more movements having short duration (e.g. when you snap your fingers)
# compared to the 'camera.framerate = 1' setting.
# In this example, "short duration" means the one which is short compared to 1 s,
# the time interval between the frames for the 'camera.framerate = 1' setting.

# Note that the 'motionDetectionThreshold' parameter (see below)
# is generally inversely proportional to the 'camera.framerate' one
# for detection of the same motion.
# 'camera.framerate = 1' setting will allow you to capture
# slower movements than 'camera.framerate = 49' setting
# for the same value of the 'motionDetectionThreshold'.

# Combining motion detection at two different framerates (e.g. 1 fps and 49 fps),
# each with a corresponding motion detection threshold,
# would allow the code to capture a wider range of both fast and slow movements
# having different durations compared to the case of a single framerate and
# motion detection threshold.

# Note that ambient light intensity variations can be captured by the measures
# of motion and cause them to exceed a motion detection threshold.
# Note therefore that the 'motionDetectionThreshold' parameter
# in the code below depends on the illumination characteristics of the scene.
# The dependence of the 'motionDetectionThreshold' on the illumination intensity
# can be calibrated for a given scene/environment.
# Statistical properties of the 'sSAD' and the 'motionEstimate' measures of motion
# for a given environment can be used to set the 'motionDetectionThreshold' too.


import numpy as np
import picamera # Please see https://picamera.readthedocs.io
				# for the 'picamera' library documentation.
                # As of April 2017, the latest 'picamera' library version is 1.13.
import picamera.array
import time
import datetime
import os


experimentDurationHours = 0.05 # Duration of the data collection, hours.
							   # 0.05 hours = 3 minutes.

timeSliceDurationMinutes = 3 # The whole 'experimentDurationHours' time
							 # is split into 'timeSliceDurationMinutes'
							 # minutes long intervals ("time slices").

experimentDir = "./experiment/" # Location where data, video, etc. will be saved.
								# Each "time slice" has its own sub-folder (see below).

motionDetectionThreshold = 1000 # Threshold for the 'motionEstimate' value (see below).
								# Motion is considered to be detected when
								# 'motionEstimate' > 'motionDetectionThreshold'.


# Motion vector data array 'a' is a 45 rows x 81 columns array for 1280x720 frame size,
# 30 rows x 41 columns array for 640x480 frame size.
# See
# https://picamera.readthedocs.io/en/release-1.13/recipes2.html#recording-motion-vector-data
# and
# https://github.com/waveform80/picamera/blob/master/picamera/array.py
# for information on the relationship between the motion vector data array size
# and the video frame size.
# "Motion data values are 4-bytes long, consisting of
# a signed 1-byte x vector, a signed 1-byte y vector,
# and an unsigned 2-byte SAD (Sum of Absolute Differences) value for each macro-block"
# (https://picamera.readthedocs.io/en/release-1.13/recipes2.html#recording-motion-vector-data).
# Therefore, the 'columnsPerRow' value should be set according to the
# 'camera.resolution = (..., ...)' setting (see below).
## rows = 45
columnsPerRow = 81

os.makedirs(experimentDir) # Error if the 'experimentDir' folder exists.

# See https://picamera.readthedocs.io/en/release-1.13/api_array.html#pimotionanalysis
# for the 'picamera.array.PiMotionAnalysis' class documentation.
class mmotion(picamera.array.PiMotionAnalysis):

    def analyse(self, a):

        global currentFrame

        analysisStartTime = time.time()

#sSAD:
        #This is the 'sSAD' value referred to in the
        # https://github.com/lvetech/ALT/blob/master/README.md file:
        sSAD = np.sum(a['sad'])
        sSADs.append(sSAD)

        # Note that both the 'sSAD' and the 'motionEstimate' values
        # for an I-frame in the captured video data stream will be equal to zero.
        # Please consult documentation for the 'start_recording()'
        # method of the 'picamera.PiCamera' class
        # (https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.start_recording).
        # Particularly, setting the 'intra_period' parameter of the 'start_recording()'
        # method to zero will cause "the encoder to produce a single initial I-frame,
        # and then only P-frames subsequently".
        # If you would like to keep I-frames in the captured video stream,
        # you can adjust the 'intra_period' parameter accordingly
        # (or leave it at its default value).
        # A very primitive way to process the I-frame 'sSAD' values
        # would be to replace them with the 'sSAD' value of the previous frame,
        # as the following 'pseudo code' shows:

        #if sSAD != 0:
            #sSADsNoZeros.append(sSAD)
        #else:
            #if len(sSADsNoZeros) >= 1:
                #sSADsNoZeros.append(sSADsNoZeros[-1])

#MOTION:
        Xabs = np.absolute(a['x']) # Calculate the absolute values of the
        						   # X-axis motion vector components

        Yabs = np.absolute(a['y']) # Calculate the absolute values of the
        						   # Y-axis motion vector components

        XabsSum = np.sum(Xabs)
        YabsSum = np.sum(Yabs)

    	# Estimate of the "amount of motion" in the frame:
        motionEstimate = XabsSum + YabsSum
        motionEstimates.append(motionEstimate)

# Uncomment for test purposes only!
        #print('motionEstimate = ', motionEstimate)

        # Note that we do not compute the lengths of the motion vectors.
        # Instead, we use a simple integral measure for the "amount of motion"
        # in the frame ('sSAD' and/or 'motionEstimate')
        # and also determine the position of the element of the motion vector
        # data array having the largest motion vector component along the X axis
        # and the one having the largest motion vector component along the Y axis (see below).


        if motionEstimate > motionDetectionThreshold:
            motionDetectionTime = time.time()
            motionDetectionTimes.append(motionDetectionTime)
            motionDetectionFrameNumbers.append(currentFrame)

        # We find the element of the motion vector data array which has the
        # motion vector with the largest X-axis component, and
        # the element of the motion vector data array which has the
        # motion vector with the largest Y-axis component below.

            indexMaxXabs = np.argmax(Xabs)
            indexMaxYabs = np.argmax(Yabs)

        # position (row, column) in the motion vector data array of the element having the
        # motion vector with the largest X-axis component:
            rowForMaxXabs = int(indexMaxXabs/columnsPerRow)
            columnForMaxXabs = indexMaxXabs - rowForMaxXabs*columnsPerRow
        # position (row, column) in the motion vector data array of the element having the
        # motion vector with the largest Y-axis component:
            rowForMaxYabs = int(indexMaxYabs/columnsPerRow)
            columnForMaxYabs = indexMaxYabs - rowForMaxYabs*columnsPerRow

            analysisStopTime = time.time()

            # The "motion detected" path
            # (the code between the 'analysisStartTime' and 'analysisStopTime', lines 115-191 above)
            # is quite fast, taking
            # less than 1 ms on a Raspberry Pi 3B unit to complete with the
            # 'camera.resolution = (1280, 720)' setting (see below).
            # Note that the 'print' statements are "slow".
            # You can gauge the duration of the 'print' statements in a way
            # similar to the one used for the "motion detected" path above.
            # Generally, comment out any 'print' statements in any part of your
            # code which is "execution time -sensitive".

# Uncomment for test purposes only!
#            print("'analyse' function run time, 'motion detected' path: ", str(analysisStopTime - analysisStartTime))
#            print('motion with amplitude ' + str(motionEstimate) + ' was detected at time ' + str(motionDetectionTime) + ' in frame ' + str(currentFrame))
#            print('MAX element indices for Xabs, Yabs: ', str(indexMaxXabs), ', ', str(indexMaxYabs))
#            print('MAX Xabs element row, column: ', rowForMaxXabs, ', ', columnForMaxXabs)
#            print('MAX Yabs element row, column: ', rowForMaxYabs, ', ', columnForMaxYabs)

        currentFrame += 1


# The relevant sections of the 'picamera' library documentation
# for the following sections of the code are:
# https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
# and
# https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.start_recording

with picamera.PiCamera() as camera:

    with mmotion(camera) as mvdOutput: # motion vector data (mvd) output

        camera.resolution = (1280, 720)
        camera.framerate = 49
        camera.exposure_mode = 'night'
        camera.awb_mode = 'auto'
        camera.iso = 1600
        camera.sharpness = 100
        camera.contrast = 100

        while camera.analog_gain <= 1:
            time.sleep(0.1)

        print('Preparing ...')
        print('30 ...')
        time.sleep(15)
        print('15 ...')
        time.sleep(5)

        # Fixing the image capture parameters can be essential
        # for the 'sSAD' measure of motion.
        # Note that relatively slow large-amplitude ambient light intensity variations
        # (e.g. on a cloudy day) can change the 'sSAD' and/or 'motionEstimate' "baseline"
        # and thus cause these measures of motion to exceed a motion detection
        # threshold when the image capture parameters are fixed before
        # the start of the video capture.
        # Note also that relatively fast ambient light intensity variations
        # can be captured by the measures of motion and cause them
        # to exceed a motion detection threshold when the variations lead to
        # significant illumination changes over the time interval between the frames.
        # Such lighting variations can be produced, for example, by incandescent
        # light bulbs (at e.g. 60 Hz in the U.S.),
        # especially if the incandescent light bulbs are the only source of light for
        # a scene observed by a camera.

        # You can experiment with the various camera settings
        # to observe their effect on the 'sSAD' and 'motionEstimate' behavior
        # for various lighting environments.

        camera.shutter_speed = camera.exposure_speed
        camera.exposure_mode = 'off'
        g = camera.awb_gains
        camera.awb_mode = 'off'
        camera.awb_gains = g
        
        print("'camera.shutter_speed' set before recording: ", camera.shutter_speed)
        print("'camera.awb_gains' set before recording: ", g)
        print("'camera.analog_gain' value before recording: ", camera.analog_gain)        

        print('10 ...')
        time.sleep(5)
        print('5 ...')
        time.sleep(5)

        print('RUNNING ...')

        for t in range(int(experimentDurationHours*60/timeSliceDurationMinutes)):

            startDateTime = datetime.datetime.now()

            timeSliceDir = experimentDir + str(startDateTime) + "/"
            print('timeSliceDir = ', timeSliceDir)
            os.makedirs(timeSliceDir)

#sSAD:
            sSADs = []
            sSADsfile = open(timeSliceDir + 'SADs.txt', 'w')

#MOTION:
            motionEstimates = []
            motionDetectionTimes = []
            motionDetectionFrameNumbers = []
            currentFrame = 1
            motionEstimateFile = open(timeSliceDir + 'motionEstimate.txt', 'w')
            motionDetectionTimesFile = open(timeSliceDir + 'motion detection times.txt', 'w')

#RECORDING:
            # Note that the 'quality' parameter of the 'start_recording()' method
            # might be useful to keep the size of the captured video files reasonably low.
            # See https://picamera.readthedocs.io/en/release-1.13/api_camera.html#picamera.PiCamera.start_recording
            # for details.
            camera.start_recording(timeSliceDir + '1280x720.h264', format = 'h264', motion_output = mvdOutput, quality = 40)
            camera.wait_recording(timeSliceDurationMinutes*60)
            camera.stop_recording()

            # Note that saving data into files and stopping/restarting video recording will cause
            # a short time "gap" between the consecutive "time slices".
#sSAD:
            for i in range(len(sSADs)):
                sSADsfile.write(str(i + 1) + ": " + str(sSADs[i]) + "\n")

#MOTION:
            for i in range(len(motionEstimates)):
                motionEstimateFile.write(str(i + 1) + ": " + str(motionEstimates[i]) + "\n")

            for i in range(len(motionDetectionTimes)):
                motionDetectionTimesFile.write(str(motionDetectionFrameNumbers[i]) + ': ' + str(motionDetectionTimes[i]) + "\n") # frame number: time


            sSADsfile.close()
            motionEstimateFile.close()
            motionDetectionTimesFile.close()

        print("'camera.shutter_speed' after the end of recording: ", camera.shutter_speed)
        print("'camera.awb_gains' after the end of recording: ", camera.awb_gains)
        print("'camera.analog_gain' value after the end of recording: ", camera.analog_gain)
        
