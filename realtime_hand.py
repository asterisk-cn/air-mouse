import time
from enum import Enum, Flag, auto

import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

mouse = Controller()

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

SENSITIVITY = 1.0

FINGER_THRESHOLD = 0.1
MOVE_THRESHOLD = 0.01

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


class HandState(Flag):
    THUMB_UP = auto()
    INDEX_UP = auto()
    MIDDLE_UP = auto()
    RING_UP = auto()
    PINKY_UP = auto()

    OPEN = THUMB_UP | INDEX_UP | MIDDLE_UP | RING_UP | PINKY_UP
    CLOSED = 0


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

    def draw_landmark(self, landmark, image):
        cv2.circle(
            image,
            (
                int(self.position[landmark.value].x * CAPTURE_WIDTH),
                int(self.position[landmark.value].y * CAPTURE_HEIGHT),
            ),
            5,
            (0, 0, 255),
            3,
        )


def get_hand_state(hand):
    state = HandState.OPEN

    if hand.get_distance(HandLandmark.THUMB_CMC, HandLandmark.THUMB_TIP) < FINGER_THRESHOLD:
        state &= ~HandState.THUMB_UP
    if hand.get_distance(HandLandmark.INDEX_FINGER_MCP, HandLandmark.INDEX_FINGER_TIP) < FINGER_THRESHOLD:
        state &= ~HandState.INDEX_UP
    if hand.get_distance(HandLandmark.MIDDLE_FINGER_MCP, HandLandmark.MIDDLE_FINGER_TIP) < FINGER_THRESHOLD:
        state &= ~HandState.MIDDLE_UP
    if hand.get_distance(HandLandmark.RING_FINGER_MCP, HandLandmark.RING_FINGER_TIP) < FINGER_THRESHOLD:
        state &= ~HandState.RING_UP
    if hand.get_distance(HandLandmark.PINKY_MCP, HandLandmark.PINKY_TIP) < FINGER_THRESHOLD:
        state &= ~HandState.PINKY_UP

    return state


def get_mouse_state(hand_state):
    if hand_state == HandState.OPEN:
        return MouseState.NONE
    elif hand_state == HandState.CLOSED & ~HandState.THUMB_UP & ~HandState.INDEX_UP:
        return MouseState.LEFT
    elif hand_state == HandState.OPEN & ~HandState.MIDDLE_UP & ~HandState.RING_UP & ~HandState.PINKY_UP:
        return MouseState.RIGHT
    elif hand_state == HandState.OPEN & ~HandState.INDEX_UP & ~HandState.MIDDLE_UP & ~HandState.RING_UP:
        return MouseState.DOUBLE
    elif hand_state == HandState.CLOSED:
        return MouseState.SCROLL
    else:
        return MouseState.NONE


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

is_moveable = False

pre_pos = Vector3(0, 0, 0)
pre_state = MouseState.NONE

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

    hand_state = get_hand_state(hand)
    mouse_state = get_mouse_state(hand_state)

    pos = hand.get_position(HandLandmark.WRIST)

    move_distance = calc_distance(pos, pre_pos)
    dx, dy = 0, 0
    if move_distance > MOVE_THRESHOLD:
        dx = int((pos.x - pre_pos.x) * CAPTURE_WIDTH * SENSITIVITY)
        dy = int((pos.y - pre_pos.y) * CAPTURE_HEIGHT * SENSITIVITY)

    if is_moveable:

        if mouse_state == MouseState.NONE:
            mouse.move(dx, dy)

        if mouse_state == MouseState.LEFT:
            mouse.press(Button.left)
        else:
            mouse.release(Button.left)

        if mouse_state == MouseState.RIGHT:
            mouse.click(Button.right)

        if mouse_state == MouseState.DOUBLE:
            mouse.click(Button.left, 2)

        if mouse_state == MouseState.SCROLL:
            mouse.scroll(0, dy)

    hand.draw_landmark(HandLandmark.WRIST, frame)

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

    cv2.putText(
        frame,
        f"pos: {pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.putText(frame, f"state: {hand_state}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"state: {mouse_state}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"move: {is_moveable}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Detection", frame)

    pre_pos = pos
    pre_state = mouse_state

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    if cv2.waitKey(10) & 0xFF == ord("z"):
        is_moveable = not is_moveable

cap.release()
cv2.destroyAllWindows()
