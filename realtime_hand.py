import time
from enum import Enum, auto

import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mouse = Controller()

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

SENSITIVITY = 1.0

# range of moving average
N_CONV = 3


class HandLandmark(Enum):
    WRIST = 0
    THUMB_CMC = auto()
    THUMB_MCP = auto()
    THUMB_IP = auto()
    THUMB_TIP = auto()
    INDEX_FINGER_MCP = auto()
    INDEX_FINGER_PIP = auto()
    INDEX_FINGER_DIP = auto()
    INDEX_FINGER_TIP = auto()
    MIDDLE_FINGER_MCP = auto()
    MIDDLE_FINGER_PIP = auto()
    MIDDLE_FINGER_DIP = auto()
    MIDDLE_FINGER_TIP = auto()
    RING_FINGER_MCP = auto()
    RING_FINGER_PIP = auto()
    RING_FINGER_DIP = auto()
    RING_FINGER_TIP = auto()
    PINKY_MCP = auto()
    PINKY_PIP = auto()
    PINKY_DIP = auto()
    PINKY_TIP = auto()


class MouseState(Enum):
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    DOUBLE = auto()
    SCROLL = auto()


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def calc_distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def calc_moving_average(landmarks):
    if len(landmarks) == 0:
        return Vector3(0, 0, 0)
    return Vector3(
        sum([landmark.x for landmark in landmarks]) / len(landmarks),
        sum([landmark.y for landmark in landmarks]) / len(landmarks),
        sum([landmark.z for landmark in landmarks]) / len(landmarks),
    )


class Hand:
    def __init__(self, n_conv):
        self.n_conv = n_conv
        self.position = [Vector3(0, 0, 0) for _ in range(len(HandLandmark))]

        self.position_history = []

    def update(self, hand_landmarks):
        if hand_landmarks is None:
            return

        tmp = [Vector3(0, 0, 0) for _ in range(len(HandLandmark))]
        for i in range(len(HandLandmark)):
            tmp[i].x = hand_landmarks.landmark[i].x
            tmp[i].y = hand_landmarks.landmark[i].y
            tmp[i].z = hand_landmarks.landmark[i].z

        self.position_history.append(tmp)
        if len(self.position_history) > self.n_conv:
            self.position_history.pop(0)

        for index in HandLandmark:
            i = index.value
            self.position[i] = calc_moving_average(
                [self.position_history[j][i] for j in range(len(self.position_history))]
            )

    def get_position(self, a=None):
        if a is None:
            return self.position
        return self.position[a.value]

    def get_distance(self, a, b):
        return calc_distance(self.position[a.value], self.position[b.value])

    def is_not_updated(self):
        return len(self.position_history) == 0

    def draw_point(self, image):
        # draw circle for each selected landmark
        if len(self.position_history) == 0:
            return

        landmark_list = [
            HandLandmark.WRIST,
            HandLandmark.THUMB_TIP,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.PINKY_TIP,
        ]

        for landmark in landmark_list:
            cv2.circle(
                image,
                (
                    int(self.position[landmark.value].x * CAPTURE_WIDTH),
                    int(self.position[landmark.value].y * CAPTURE_HEIGHT),
                ),
                5,
                (0, 0, 255),
                -1,
            )

        cv2.circle(
            image,
            (
                int(self.position[HandLandmark.INDEX_FINGER_MCP.value].x * CAPTURE_WIDTH),
                int(self.position[HandLandmark.INDEX_FINGER_MCP.value].y * CAPTURE_HEIGHT),
            ),
            10,
            (0, 0, 255),
            3,
        )


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap_fps = cap.get(cv2.CAP_PROP_FPS)


hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    max_num_hands=1,
)

pre_pos = Vector3(0, 0, 0)

hand = Hand(N_CONV)


while cap.isOpened():
    prev_time = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        assert len(results.multi_hand_landmarks) == 1

        hand_landmarks = results.multi_hand_landmarks[0]
        hand.update(hand_landmarks)
        hand.draw_point(frame)

    pos = hand.get_position(HandLandmark.INDEX_FINGER_MCP)
    dx = pos.x - pre_pos.x
    dy = pos.y - pre_pos.y

    dX = int(dx * CAPTURE_WIDTH * SENSITIVITY)
    dY = int(dy * CAPTURE_HEIGHT * SENSITIVITY)

    threshold = 0.1
    if abs(dX) > threshold or abs(dY) > threshold:
        mouse.move(dX, dY)

    cv2.putText(
        frame,
        f"cameraFPS: {cap_fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    process_fps = 1 / (time.perf_counter() - prev_time)
    cv2.putText(
        frame,
        f"processFPS: {process_fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    pos = hand.get_position(HandLandmark.INDEX_FINGER_MCP)
    cv2.putText(
        frame,
        f"pos: {pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Hand Detection", frame)

    pre_pos = pos

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
