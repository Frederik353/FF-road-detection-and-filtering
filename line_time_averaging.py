import numpy as np
from typing import Tuple
import constants

# Assuming constants are defined in the constants module, e.g.:
# constants.BUFFER_SIZE, constants.LEFT_FALLBACK_LINE, constants.RIGHT_FALLBACK_LINE, constants.ALPHA

line_buffer = np.array([[constants.LEFT_FALLBACK_LINE, constants.RIGHT_FALLBACK_LINE]] * constants.BUFFER_SIZE)

WEIGHTS = np.array([[constants.ALPHA ** (constants.BUFFER_SIZE - 1 - i)] for i in range(constants.BUFFER_SIZE)])
NORMALIZED_WEIGHTS = WEIGHTS / WEIGHTS.sum()


def add_line_to_buffer(left_line: np.ndarray, right_line: np.ndarray) -> None:
    """
    Adds a new pair of lines to the buffer, maintaining the buffer size by removing the oldest entry if necessary.

    Parameters:
    - left_line (np.ndarray): The coordinates for the new left line.
    - right_line (np.ndarray): The coordinates for the new right line.
    """
    global line_buffer
    # Append the new lines to the buffer and ensure the buffer size doesn't exceed its limit
    line_buffer = np.append(line_buffer[-constants.BUFFER_SIZE + 1 :], [[left_line, right_line]], axis=0)


def calculate_EMWA_lines() -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Exponentially Moving Weighted Average (EMWA) for left and right lines in the buffer.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the averaged left and right line coordinates.
    """
    global line_buffer, NORMALIZED_WEIGHTS

    # Calculate the weighted average for left and right lines
    avg_left = np.sum(NORMALIZED_WEIGHTS * line_buffer[:, 0, :], axis=0)
    avg_right = np.sum(NORMALIZED_WEIGHTS * line_buffer[:, 1, :], axis=0)

    return avg_left, avg_right
