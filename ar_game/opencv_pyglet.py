import random
import cv2
import numpy as np
import pyglet
from PIL import Image
import sys
import cv2.aruco as aruco
import os


NET_WIDTH = 128
NET_HEIGHT = NET_WIDTH
video_id = 0
points = []
fingertip = None
score = 0
lives = 3

# image from https://www.svgrepo.com/svg/321919/bug-net?edit=true
net_path = os.path.normpath("./bugnet.png")
# image from https://www.svgrepo.com/svg/500097/butterfly?edit=true
butterfly_path = os.path.normpath("./butterfly.png")


if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()


# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# https://stackoverflow.com/questions/39953263/get-video-dimension-in-python-opencv
res_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
res_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

WINDOW_WIDTH = res_width
WINDOW_HEIGHT = res_height

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

batch = pyglet.graphics.Batch()

score_box = pyglet.shapes.Rectangle(0, WINDOW_HEIGHT-50, 100, 50, color=(200, 200, 200), batch=batch)
score_label = pyglet.text.Label(text=f"Score: {score}", x=10, y=WINDOW_HEIGHT-30, batch=batch)

end_label = pyglet.text.Label(text=f"Your Score: {score} press 'r' to restart or 'q' to exit", x=WINDOW_WIDTH/4,
                              y=WINDOW_HEIGHT/2)


# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img, fmt):
    # Assumes image is in BGR color space. Returns a pyimg object
    if fmt == 'GRAY':
        rows, cols = img.shape
        channels = 1
    else:
        rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels * cols
    pyimg = pyglet.image.ImageData(width=cols,
                                   height=rows,
                                   fmt=fmt,
                                   data=raw_img,
                                   pitch=top_to_bottom_flag * bytes_per_row)
    return pyimg


def warp(input_img, width, height):
    # https://youtu.be/WQeoO7MI0Bs?t=2778
    p1 = np.float32([points[0], points[1], points[3], points[2]])
    p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(p1, p2)
    output = cv2.warpPerspective(input_img, matrix, (width, height))

    return output


class Net:
    global net_path

    def __init__(self, x=50, y=50, img=net_path, scale=0.5):
        self.x = x
        self.y = y
        self.img = pyglet.image.load(img)
        self.net = pyglet.sprite.Sprite(self.img, x=x, y=y, batch=batch)
        self.net.scale = scale

    def set_pos(self, x, y):
        self.x = x
        self.y = y
        self.net.position = (self.x, self.y, 0)

    def draw(self):
        self.net.draw()


net = Net()


class Enemy:
    enemies = []

    def __init__(self, x, y, img=butterfly_path, radius=30, scale=0.2):
        self.x = x
        self.y = y
        self.radius = radius
        self.img = pyglet.image.load(img)
        self.butterfly = pyglet.sprite.Sprite(self.img, x=x, y=y, batch=batch)
        self.butterfly.scale = scale

    def update_enemies():
        for enemy in Enemy.enemies:
            enemy.move()

    def draw_enemies():
        for enemy in Enemy.enemies:
            enemy.draw()

    @staticmethod
    def create_enemy(x, y):
        Enemy.enemies.append(Enemy(x, y))

    def move(self):
        self.x -= 5
        self.butterfly.position = (self.x, self.y, 0)

    def draw(self):
        self.butterfly.draw()

    def delete_enemy(self):
        if self in Enemy.enemies:
            Enemy.enemies.remove(self)
            self.butterfly.delete()
            print("delete")

    @staticmethod
    def collision_detection(tip):
        global score
        if tip is not None:
            for enemy in Enemy.enemies:
                # Exercise sheet one
                dist_x = tip[0] - enemy.x
                dist_y = (res_height - tip[1]) - enemy.y
                dist = np.sqrt(pow(dist_x, 2) + pow(dist_y, 2))
                print(dist)
                if dist <= enemy.radius:
                    enemy.delete_enemy()
                    score += 1
                    score_label.text = f"Score: {score}"

    def out_of_bounds():
        global lives
        for enemy in Enemy.enemies:
            if enemy.x < 0:
                enemy.delete_enemy()
                lives -= 1


def spawn_enemies():
        rand_y = random.randint(0, WINDOW_HEIGHT)
        rand = random.randint(0, 60)
        if rand == 0 or rand == 50:
            Enemy.create_enemy(WINDOW_WIDTH, rand_y)
        if len(Enemy.enemies) > 0:
            Enemy.update_enemies()


def detect_markers(frame):
    global points
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Check if marker is detected
    if ids is not None:
        # Draw lines along the sides of the marker
        aruco.drawDetectedMarkers(frame, corners)
        if len(ids) == 4:

            # chatGPT to find out how to sort the markers
            id_corner_map = {}
            for i in range(len(ids)):
                id_corner_map[ids[i][0]] = corners[i][0]
            # Sort the corners based on the marker IDs
            sorted_corners = []
            for i in range(4):
                sorted_corners.append(id_corner_map[i])

            # Reset the points list
            points = []
            for corner in sorted_corners:
                points.append(corner[0])

            frame = warp(frame, WINDOW_WIDTH, WINDOW_HEIGHT)

            return frame


def get_finger(frame):
    global fingertip

    kernel = np.ones((5, 5), np.uint8)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find pixel with highest x value in contours and in inner frame | chatGPT to find max x in contours
    max_x = 0
    for contour in contours:
        for point in contour:
            x, y = point[0]
            # without borders --> only checks contours in the frame
            if 30 < x < res_width - 30 and 30 < y < res_height - 30:
                if x > max_x:
                    max_x = x
                    fingertip = (x, y)

    if fingertip is not None:
        # position net
        net.set_pos(fingertip[0] - NET_WIDTH/2, (res_height - fingertip[1]) - NET_HEIGHT/2)
        # check if net is in range of butterfly
        Enemy.collision_detection(fingertip)


# from exercise sheet 2
@window.event
def on_key_press(symbol, modifiers):
    global lives
    global score
    if symbol == pyglet.window.key.Q:
        window.close()
    if symbol == pyglet.window.key.R:
        lives = 3
        Enemy.enemies = []
        score = 0
        score_label.text = f"Score: {score}"


@window.event
def on_draw():
    global lives
    window.clear()
    if lives > 0:
        ret, frame = cap.read()

        # warp frame for "game"-view
        warped_frame = detect_markers(frame)
        if warped_frame is not None:
            # if game view -> spawn, update.. butterflies
            spawn_enemies()
            frame = warped_frame
            Enemy.update_enemies()
            Enemy.out_of_bounds()
            get_finger(frame)

        img = cv2glet(frame, 'BGR')
        img.blit(0, 0, 0)

        batch.draw()
    else:
        end_label.draw()


pyglet.app.run()
