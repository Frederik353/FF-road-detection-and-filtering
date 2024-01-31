import cv2
import os
import numpy as np
import time

from filter_da import filter_da

# t = time.time()
# te = time.time()
# print(f"time {te - t}")


class debug_filter_da:
    image_range = (17, 20)

    save_result = False
    save_folder = "final_images"
    # BGR colors for each mask
    colors = [
        (237, 193, 105, 255),
        (255, 255, 255, 255),
        (157, 251, 177, 255),
        (155, 155, 155, 255),
    ]  # BGR format !!!not RGB!!!

    wait_time = 5_000  # ms between showing images

    processed_masks = []

    def __init__(self):
        time_arr = []
        for i in range(self.image_range[0], self.image_range[1]):
            print(i, "-----------------------------------------")
            try:
                # read mask from npz file
                file_path = f"./iteration/ ({i}).npz"
                npzfile = np.load(file_path)
                ll_seg_mask = npzfile["ll_seg_mask"]
                da_seg_mask = npzfile["da_seg_mask"]

                # run filter on mask
                start_time = time.time()
                masks = filter_da(da_seg_mask, ll_seg_mask)
                end_time = time.time()
                time_arr.append(start_time - end_time)
                print(f"Elapsed time: {start_time - end_time} seconds")

                image = self.combine_masks(masks)
                self.display_image(image)
                print("image", image.dtype, image.shape)

                if self.save_result:
                    self.save_image(image, i)

            except FileNotFoundError:
                print("file not found ", i)
        cv2.destroyAllWindows()

        print("----------------------------------------")
        print(f"average time: {np.average(time_arr)}")
        print(f"median time: {np.median(time_arr)}")
        print(time_arr)

    def combine_masks(self, masks):
        if len(masks) == 1:
            return masks

        img_height, img_width = masks[0].shape
        # Create an empty canvas with 4 channels (for BGR alpha) alpha channel is for transparency
        n_channels = 4
        overlay = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

        # set background to transparent
        overlay[masks[0] == 0] = (0, 0, 0, 0)
        for mask, color in zip(masks, self.colors):
            overlay[mask == 1] = color

        return overlay

    def display_image(self, image, window_name="Images"):
        """show an image

        Args:
            image (np.array): image to show
            window_name (str, optional): name of window to show image in. Defaults to "Images".
        """

        cv2.imshow(window_name, image)
        cv2.waitKey(self.wait_time)

    def save_image(self, image, filname):
        """
        Save an image to a specified folder with a given filename.
        :param image_data: numpy array containing image data to save
        """
        # so you can use int

        filename = f"{filename}.png"
        folder_name = str(self.save_folder)

        # Use os.path.join to make the code platform-independent
        folder_path = os.path.join(os.getcwd(), folder_name)

        # Check if the directory exists
        if not os.path.exists(folder_path):
            # If the directory does not exist, create it
            os.makedirs(folder_path)

        # Full path where you want to save the image
        save_path = os.path.join(folder_path, filename)

        # Save the image
        success = cv2.imwrite(save_path, image)

        # Check if the image was saved successfully
        if success:
            print(f"Image saved successfully at: {save_path}")
        else:
            print(f"Failed to save image at: {save_path}")


def debug_image(masks, palette=None, is_demo=False, t=10_000, window="debug"):
    """show an image
    Args:
        image (_type_): image to show
        t (int): time to show image in ms
    """
    # if not given define
    if palette == None:
        palette = generate_distinct_colors(len(masks))
    else:
        # if colors are given make sure you have enough for every mask
        assert len(palette) == len(masks)

    np_image = np.zeros((masks[0].shape[0], masks[0].shape[1], 3), dtype=np.uint8)

    for label, color in enumerate(palette):
        np_image[masks[label] == 1, :] = color

    # # If image is boolean, we need to convert to 0s and 255s
    # if np.max(image) == 1:
    #     np_image = image.astype(np.uint8) * 255
    # else:
    #     np_image = image.astype(np.uint8)

    # display over image
    # color_mask = np.mean(color_seg, 2)
    # img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

    print("showing debug image")
    cv2.imshow(window, np_image)
    cv2.waitKey(t)  # Display the image for 10 seconds
    # cv2.destroyAllWindows()

    # test_image = np.zeros_like(ll_seg_mask)
    # for line in averaged_lines+averaged_lines2:
    #     x1, y1, x2, y2 = line  # hvis input averaged lines
    #     cv2.line(test_image, (x1, y1), (x2, y2), (1, 0, 0), 1)
    # debug_image([test_image])

    # print("lines", averaged_lines)


def generate_distinct_colors(n):
    """
    Generate a list of n distinct colors.

    This function generates distinct colors by evenly sampling
    the hue component in the HSV color space and then converting
    them to the RGB color space.

    Args:
    n (int): The number of distinct colors to generate.

    Returns:
    np.array: A list of RGB colors, each represented as a tuple of three integers.
    """

    # Generate colors in HSV space. HSV is used because varying the hue
    # with a fixed saturation and value gives good color diversity.
    # OpenCV's Hue range is from 0-180 (instead of 0-360), hence the scaling.
    hsv_colors = [(i * 180 / n, 255, 255) for i in range(n)]

    # Convert HSV colors to RGB
    rgb_colors = np.array([cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0] for hsv in hsv_colors])
    return rgb_colors


# for making discontinous lines if you dont have good examples to test on
def tamper(image, block_size, shift_amount):
    """create a tampered image by shifting blocks of rows in the image

    Args:
        image (_type_):
        block_size (_type_): size of blocks being shifted
        shift_amount (_type_): amount of pixels to shift left and right

    Returns:
        _type_: tampered image with discontinous lines

    example: use
        # copy = shift_outwards_and_fill(np_image, 50, 50)
    """

    rows, cols = image.shape
    output_image = np.copy(image)

    for block_start in range(0, rows, block_size * 2):
        for i in range(block_start, min(block_start + block_size, rows)):
            # Shift the row by the shift_amount
            output_image[i, shift_amount:] = output_image[i, :-shift_amount]
            # Fill the start of the row with zeros
            output_image[i, cols - shift_amount :] = 0

    mid_col = cols // 2
    output_image[:, :mid_col] = np.fliplr(output_image[:, mid_col:])

    return output_image


if __name__ == "__main__":
    foo = debug_filter_da()

# old
# average time: -0.02864196565416124
# median time: -0.021003007888793945

# new
# average time: -0.024868737326727973
# median time: -0.022040367126464844
