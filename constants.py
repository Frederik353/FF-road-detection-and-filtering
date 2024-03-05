import numpy as np

# Parameters for cv2.HoughLinesP

# RHO: Distance resolution of the accumulator in pixels.
# This parameter determines the precision of the line detection. A smaller RHO means higher resolution
# and the ability to detect more subtle line distinctions. However, too small a value can lead to noise
# and false positives. Adjust this value based on the level of detail and noise in your image.
RHO = 2

# THETA: Angular resolution of the accumulator in radians.
# This defines the precision of the angle at which lines are detected. A smaller THETA value
# increases the angular resolution. Adjust this value to be able to detect lines at varying angles,
# especially if your application requires detecting lines not strictly horizontal or vertical.
THETA = np.pi / 180  # 1 degree

# THRESHOLD: The minimum number of intersections in a grid cell for a line to be considered valid.
# Increase this value to reduce false positives, i.e., to ensure that only lines with significant
# evidence are detected. However, setting this too high can cause missed detections in weaker or fragmented lines.
THRESHOLD = 100

# MIN_LINE_LENGTH: The minimum number of points that can form a line. Lines with fewer points are discarded.
# Increase this value to detect longer, more continuous lines while ignoring short, fragmented segments.
# Decrease it if you need to detect shorter lines.
MIN_LINE_LENGTH = 20

# MAX_LINE_GAP: The maximum gap between segments that will be connected to form a single line.
# Increase this to allow disconnected line segments to be joined into a single line, which can be useful
# in situations where the line visibility is intermittent. Decrease it to prevent the merging of separate line segments.
MAX_LINE_GAP = 50

# Parameters for exponential moving weighted average (EMWA)

# BUFFER_SIZE: Determines how many past observations are considered in the weighted average.
# A larger buffer incorporates more history into the average, resulting in smoother line estimations
# but can delay the detection of changes. A smaller buffer responds more quickly to changes but can result in jittery line estimations.
BUFFER_SIZE = 3

# ALPHA: The decay factor for past observations in the EMWA.
# A value closer to 1.0 gives more weight to recent observations, making the line estimations more responsive to sudden changes.
# A value closer to 0 spreads the weight more evenly across the buffer, leading to smoother but potentially less responsive line estimations.
ALPHA = 0.0

# Fallback lines
# These are used as default values when the line detection algorithm fails to detect lines.
# Ideally, these values should be representative of the typical line positions and orientations in your application.
# You may need to adjust these based on the average slope and y-intercept values observed in successful detections.
RIGHT_FALLBACK_LINE = (0.17578125, 155)  # Slope and y-intercept for the fallback right line.
LEFT_FALLBACK_LINE = (-0.18046875, 406)  # Slope and y-intercept for the fallback left line.

# CC_MAX_DISTANCE: The maximum distance to consider when connecting line segments.
# Increasing this value allows the algorithm to bridge wider gaps between segments, which can be useful in low-quality images or
# where lines are frequently broken. However, setting this too high can lead to incorrect connections between unrelated line segments.
CC_MAX_DISTANCE = 100
