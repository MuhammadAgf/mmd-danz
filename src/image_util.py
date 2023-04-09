from typing import Optional, Tuple

import cv2
import numpy as np


def crop_image(img, vectors):
    center_x = np.mean(vectors[:, 0])
    center_y = np.mean(vectors[:, 1])
    center = (center_x, center_y)

    crop_size = (700, 700)
    # Compute the bounding box for the cropped region
    x, y = int(center[0] - crop_size[0] / 2), int(center[1] - crop_size[1] / 2)
    w, h = crop_size

    # Adjust the bounding box to keep the center point the same
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > img.shape[1]:
        x = img.shape[1] - w
    if y + h > img.shape[0]:
        y = img.shape[0] - h

    # Extract the cropped region from the image
    img_cropped = cv2.getRectSubPix(img, (w, h), (x + w / 2, y + h / 2))
    return img_cropped


def crop_unstable(image, keypoints, ratio):
    x_min = int(min(keypoints[:, 0]) - int(image.shape[0] * ratio))
    x_max = int(max(keypoints[:, 0]) + int(image.shape[1] * ratio))
    y_min = int(min(keypoints[:, 1]) - int(image.shape[0] * ratio))
    y_max = int(max(keypoints[:, 1]) + int(image.shape[1] * ratio))

    # Crop the image to the bounding box of the keypoints
    cropped_image = image[
        max(y_min, 0) : min(y_max, image.shape[0] - 1),
        max(x_min, 0) : min(x_max, image.shape[1] - 1),
    ]
    return cropped_image


def resize_and_concat_images(image, reference_image, ratio=0.08, flip=False):
    # Find the bounding box of the keypoints
    if len(keypoints) > 0:
        # cropped_image = crop_unstable(image, keypoints, ratio)
        cropped_image = crop_image(image, keypoints)
    else:
        cropped_image = crop_image(
            image, np.array([[image.shape[0] // 2, image.shape[1] // 2]])
        )
    # Compute the scale factor based on the height of the reference image
    scale_factor = reference_image.shape[0] / cropped_image.shape[0]

    # Resize the cropped image vertically while maintaining the aspect ratio
    resized_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor)

    if flip:
        resized_image = cv2.flip(resized_image, 1)
    # Concatenate the resized image and the reference image vertically
    result_image = cv2.hconcat([resized_image, reference_image])

    return result_image


def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line, font_face, font_scale, get_text_size_font_thickness
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[y : y + sz_h, x : x + sz_w] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb
