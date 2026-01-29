import cv2
import numpy as np


def grayscale(img):
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Apply Canny edge detection."""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """Apply a mask to keep only the region of interest."""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def draw_lines(img, lines, color=(0, 255, 0), thickness=2):
    """Draw lines on the image."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Detect lines using Hough Transform."""
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


def process_image(image):
    """Process the image to detect lane lines."""
    gray = grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = canny(blurred, 50, 150)

    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, roi_vertices)

    lines = hough_lines(roi, 1, np.pi / 180, 50, 100, 150)

    result = image.copy()
    draw_lines(result, lines)

    return result
