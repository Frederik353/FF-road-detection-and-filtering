import numpy as np


# todo write parameter explinations
# todo tune parameters
# Parameters for cv2.HoughLinesP
RHO = 2  # Distance resolution of the accumulator in pixels
THETA = np.pi / 180  # Angular resolution of the accumulator in radians (1 degree)
THRESHOLD = 100  # Threshold: minimum number of intersections to detect a line
MIN_LINE_LENGTH = 20  # Minimum length of a line (in pixels) to be accepted
MAX_LINE_GAP = 50  # Maximum gap between points on the same line to link them


BUFFER_SIZE = 3
ALPHA = 0.0


# todo find better guess, either by a better avg or by other technique
# RIGHT_FALLBACK_LINE = [1280, 380, 0, 155]
# LEFT_FALLBACK_LINE = [0, 406, 1280, 175]
RIGHT_FALLBACK_LINE = (0.17578125, 155)
LEFT_FALLBACK_LINE = (-0.18046875, 406)

# connect components max distance
CC_MAX_DISTANCE = 100
