import cv2
import numpy as np


def roi(image, vertices):
    mask = np.zeros_like(image)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    cropped_img = cv2.bitwise_and(image, mask)
    return cropped_img


def draw_lines(image, hough_lines):
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return image

def gaussian_blur(gray_scaled_image):
    blur = cv2.GaussianBlur(gray_scaled_image, (5, 5), 0)
    return blur

def canny_img(blured_image):
    canny = cv2.Canny(blured_image, 130, 220)
    return canny

def grayscale(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.dilate(gray_img, kernel=np.ones((3, 3), np.uint8))

    return gray_img

def hough_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=10, minLineLength=15, maxLineGap=2)

    return lines

def process(img):
    height = img.shape[0]
    width = img.shape[1]
    roi_vertices = [
        (0, 650),
        (2*width/4.2, 2*height/3),
        (width, 1000)
    ]

    gray_img = grayscale(img)

    blur = gaussian_blur(gray_img)
    canny = canny_img(blur)

    roi_img = roi(canny, np.array([roi_vertices], np.int32))

    lines = hough_lines(roi_img)

    final_img = draw_lines(img, lines)

    return final_img


cap = cv2.VideoCapture("./lane_vid2.mp4")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
saved_frame = cv2.VideoWriter("lane_detection.avi", fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    try:
        frame = process(frame)

        saved_frame.write(frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    except Exception:
        break

cap.release()
saved_frame.release()
cv2.destroyAllWindows()

