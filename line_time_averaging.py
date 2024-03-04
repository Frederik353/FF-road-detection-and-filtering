import numpy as np
import constants

line_buffer = []
# Calculate weights using exponential decay
# The most recent line has the highest weight and oldest the lowest
WEIGHTS = np.array([[constants.ALPHA ** (constants.BUFFER_SIZE - 1 - i)] for i in range(constants.BUFFER_SIZE)])
# Normalize weights so they sum to 1
NORMALIZED_WEIGHTS = WEIGHTS / WEIGHTS.sum()

# prevents errors in the first few frames and should only take a second to fill up with real values
for i in range(constants.BUFFER_SIZE):
    line_buffer.append([constants.LEFT_FALLBACK_LINE, constants.RIGHT_FALLBACK_LINE])


def add_line_to_buffer(left_line, right_line):
    global line_buffer
    lines = [left_line, right_line]
    line_buffer.append(lines)
    line_buffer = line_buffer[-constants.BUFFER_SIZE :]


def calculate_EMWA_lines():
    global line_buffer, NORMALIZED_WEIGHTS
    # Convert line_buffer to NumPy array for vectorized operations
    line_buffer_np = np.array(line_buffer)  # Shape: [number_of_lines, 2 (left, right), 2 (slope, intercept)]
    # Ensure NORMALIZED_WEIGHTS is correctly shaped for broadcasting
    # NORMALIZED_WEIGHTS should be reshaped to [number_of_lines, 1, 1] to broadcast over the line_buffer_np dimensions
    # weights = NORMALIZED_WEIGHTS[:, np.newaxis, np.newaxis]
    weights = NORMALIZED_WEIGHTS

    # Calculate the weighted average for left and right lines
    avg_left = np.sum(weights * line_buffer_np[:, 0, :], axis=0)
    avg_right = np.sum(weights * line_buffer_np[:, 1, :], axis=0)

    return avg_left, avg_right


# def calculate_EMWA_lines():
#     global line_buffer
#     # EMWA = Exponential Moving Weighted Average
#     # Initialize sums for weighted averages
#     avg_left = [0, 0]
#     avg_right = [0, 0]

#     # Calculate the weighted average for left and right lines
#     for i, ((left_line, right_line), weight) in enumerate(zip(line_buffer, constants.NORMALIZED_WEIGHTS)):
#         avg_left = [avg + (val * weight) for avg, val in zip(avg_left, left_line)]
#         avg_right = [avg + (val * weight) for avg, val in zip(avg_right, right_line)]

#     # avg_left = [int(round(val)) for val in avg_left]
#     # avg_right = [int(round(val)) for val in avg_right]

#     return avg_left, avg_right
