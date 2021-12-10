import math
from typing import Any, List

import cv2
import numpy as np


def ensure_gray(image: np.ndarray) -> np.ndarray:
    """Ensure input image is always gray. """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def ensure_optimal_square(image: np.ndarray) -> np.ndarray:
    assert image is not None, image
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    output_image = cv2.copyMakeBorder(src=image, top=0, bottom=nh - image.shape[0],
                                      left=0, right=nw - image.shape[1],
                                      borderType=cv2.BORDER_CONSTANT, value=0)
    return output_image


def theta_to_angle(theta: float) -> float:
    # Calculate tilt angle, convert radian to celcius.
    angle = math.atan(theta) / np.pi * 180
    if angle < -45:
        angle = angle + 90
    elif angle > 45:
        angle = angle - 90
    return angle


def fft_deskew_image(image: np.ndarray, debug: bool = False, pi_thresh: float = None) -> float:
    def get_line_length(line: List[Any]) -> float:
        x1, y1, x2, y2 = line[0]
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)
        return np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)

    assert isinstance(image, np.ndarray), image
    if pi_thresh is None:
        pi_thresh = np.pi / 180

    gray = ensure_gray(image)

    # DFT in OpenCV requires the size diviable for 2, 3 and 5. and square
    opt_gray = ensure_optimal_square(gray)
    opt_gray = cv2.resize(opt_gray, (1500, 1500))
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)
    opt_gray = cv2.dilate(opt_gray, np.ones((5, 5)))

    # Perform Fourier transform
    dft = np.fft.fft2(opt_gray)  # spatial domain to the frequency domain

    # Shift the zero-frequency component to the center of the spectrum
    shifted_dft = np.fft.fftshift(dft)

    # Use abs() to get the real number (imag() to get the imaginary part),
    magnitude = np.abs(shifted_dft)

    # And the logarithm is used to transform the data to 0-255,
    # which is equivalent to achieving normalization.
    magnitude = np.log(magnitude)
    magnitude = magnitude.astype(np.uint8)

    # Binarization, Houge straight line detection
    bin_thesh = 11
    bin_magnitude = cv2.threshold(
        magnitude, thresh=bin_thesh, maxval=255, type=cv2.THRESH_BINARY)[1]
    density_value = (np.sum(bin_magnitude) /
                     bin_magnitude.shape[0] / bin_magnitude.shape[0])
    while density_value > 2.2:
        bin_thesh += 1
        bin_magnitude = cv2.threshold(
            magnitude, thresh=bin_thesh, maxval=255, type=cv2.THRESH_BINARY)[1]
        density_value = (np.sum(bin_magnitude) /
                         bin_magnitude.shape[0] / bin_magnitude.shape[0])

    # Hough lines.
    lines = cv2.HoughLinesP(bin_magnitude, rho=1, theta=pi_thresh, threshold=50,
                            minLineLength=opt_gray.shape[0] // 4, maxLineGap=100,)
    if lines is None:
        return 0

    lines = sorted(lines.tolist(), key=get_line_length, reverse=True)
    sum_angle = 0.0
    count = 0
    for line in lines[:5]:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:  # Vertical.
            continue
        elif y2 - y1 == 0:  # Horizontal.
            continue

        theta = (y2 - y1) / (x2 - x1)
        unit_angle = theta_to_angle(theta)
        if abs(unit_angle) < 0.1:
            continue

        sum_angle += unit_angle
        count += 1

    if count == 0:
        skew_angle = 0
    else:
        skew_angle = sum_angle / count
    return skew_angle


def cv_deskew_image(img: np.ndarray) -> float:
    """Adapted from: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/"""

    # Use medianBluer to remove black artefacts from the image
    image = cv2.medianBlur(img, 5)

    # Convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # Threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # The `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # Otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    return angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image to deskew it."""
    hh, ww = image.shape[:2]
    center = (ww // 2, hh // 2)
    mm = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, mm, (ww, hh), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    return rotated_image
