import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Path to input file')
parser.add_argument('output_file', type=str, help='Path to output file')
parser.add_argument('output_width', type=int, help='Width of output file as int')
parser.add_argument('output_height', type=int, help='Height of output file as int')

args = parser.parse_args()

input_file = os.path.normpath(args.input_file)
output_file = os.path.normpath(args.output_file)
output_width = int(args.output_width)
output_height = int(args.output_height)

img = cv2.imread('sample_image.jpg')
img2 = img.copy()
WINDOW_NAME = 'Preview Window'


cv2.namedWindow(WINDOW_NAME)

points = []
mouse_active = True


# click order has to be: top left, top right, bottom right and bottom left. (clockwise)
def warp(input_img, width, height):
    # https://youtu.be/WQeoO7MI0Bs?t=2778
    p1 = np.float32([points[0], points[1], points[3], points[2]])
    p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(p1, p2)
    output = cv2.warpPerspective(input_img, matrix, (width, height))

    return output


def mouse_callback(event, x, y, flags, param):
    global img
    global img2
    global points
    global mouse_active

    if mouse_active:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            img2 = cv2.circle(img2, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(WINDOW_NAME, img2)

            if len(points) == 4:
                mouse_active = False
                img_warped = warp(img, output_width, output_height)
                cv2.imshow(WINDOW_NAME, img_warped)
                save(img_warped)
                points = []


def save(img_warped):
    # Check for "s"-key
    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite(str(output_file), img_warped)
        print(f"Das Bild wurde erfolgreich gespeichert! Du findest es unter dem Namen {str(output_file)}")


cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
cv2.imshow(WINDOW_NAME, img)

while True:
    # Check for ESC-key
    if cv2.waitKey(1) == 27:
        img2 = img.copy()
        cv2.imshow(WINDOW_NAME, img2)
        points = []
        mouse_active = True

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

