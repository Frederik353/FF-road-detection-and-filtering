import numpy as np
import cv2
import timeit

# Define your binary mask here
binary_mask = np.random.randint(0, 2, (1000, 1000), dtype=np.uint8)


# Define the NumPy method as a function
def numpy_method(mask):
    flipped_mask = np.flipud(mask)
    position = np.unravel_index(np.argmax(flipped_mask), flipped_mask.shape)
    original_position = (mask.shape[0] - 1 - position[0], position[1])
    mask[original_position] = 2
    return mask


# Define the OpenCV method as a function
def opencv_method(mask):
    flipped_mask = cv2.flip(mask, 0)
    locations = cv2.findNonZero(flipped_mask)
    if locations is not None:
        first_location = locations[0][0]
        original_position = (mask.shape[0] - 1 - first_location[1], first_location[0])
        mask[original_position] = 2
    return mask


# Benchmark the methods
numpy_time = timeit.timeit("numpy_method(binary_mask.copy())", globals=globals(), number=100)
opencv_time = timeit.timeit("opencv_method(binary_mask.copy())", globals=globals(), number=100)

print(f"NumPy Time: {numpy_time}")
print(f"OpenCV Time: {opencv_time}")
