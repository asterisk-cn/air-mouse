import time
import tkinter as tk
from enum import Enum, Flag, auto

import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

SENSITIVITY = 1.0

FINGER_THRESHOLD = 0.1
MOVE_THRESHOLD = 0.01

# range of moving average
N_CONV = 3

g_is_activate = False


class HandLandmark(Enum):
    """Enum for hand landmark index"""

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
    """Enum for hand state"""

    THUMB_UP = auto()
    INDEX_UP = auto()
    MIDDLE_UP = auto()
    RING_UP = auto()
    PINKY_UP = auto()

    OPEN = THUMB_UP | INDEX_UP | MIDDLE_UP | RING_UP | PINKY_UP
    CLOSED = 0


class MouseState(Enum):
    """Enum for mouse state"""

    NONE = auto()
    NORMAL = auto()
    LEFT = auto()
    RIGHT = auto()
    DOUBLE = auto()
    SCROLL = auto()


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Config:
    def __init__(self):
        self.sensitivity = SENSITIVITY

        self.modify()

    def modify(self):
        global g_is_activate
        g_is_activate = False

        root = tk.Tk()

        root.protocol("WM_DELETE_WINDOW", exit)
        root.title("Config")
        root.geometry("370x320")
        _sensitivity = tk.IntVar()
        _sensitivity.set(self.sensitivity)

        # Sensitivity
        label = tk.Label(text="Sensitivity")
        label.pack()
        scale = tk.Scale(
            root,
            variable=_sensitivity,
            from_=0.1,
            to=10,
            resolution=0.1,
            length=300,
            orient="h",
        )
        scale.pack()
        # continue
        button = tk.Button(text="continue", command=root.destroy)
        button.pack()
        # wait
        root.mainloop()
        # output
        self.sensitivity = _sensitivity.get()


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
        self.state = HandState.OPEN

        self.position_history = []

    def update(self, hand_landmarks):
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

        self.compute_hand_state()

    def calc_distance_landmark(self, a, b):
        return calc_distance(self.position[a.value], self.position[b.value])

    def compute_hand_state(self):
        state = HandState.OPEN

        if self.calc_distance_landmark(HandLandmark.THUMB_TIP, HandLandmark.PINKY_MCP) < self.calc_distance_landmark(
            HandLandmark.THUMB_MCP, HandLandmark.PINKY_MCP
        ):
            state &= ~HandState.THUMB_UP
        if self.calc_distance_landmark(HandLandmark.INDEX_FINGER_TIP, HandLandmark.WRIST) < self.calc_distance_landmark(
            HandLandmark.INDEX_FINGER_DIP, HandLandmark.WRIST
        ):
            state &= ~HandState.INDEX_UP
        if self.calc_distance_landmark(
            HandLandmark.MIDDLE_FINGER_TIP, HandLandmark.WRIST
        ) < self.calc_distance_landmark(HandLandmark.MIDDLE_FINGER_DIP, HandLandmark.WRIST):
            state &= ~HandState.MIDDLE_UP
        if self.calc_distance_landmark(HandLandmark.RING_FINGER_TIP, HandLandmark.WRIST) < self.calc_distance_landmark(
            HandLandmark.RING_FINGER_DIP, HandLandmark.WRIST
        ):
            state &= ~HandState.RING_UP
        if self.calc_distance_landmark(HandLandmark.PINKY_TIP, HandLandmark.WRIST) < self.calc_distance_landmark(
            HandLandmark.PINKY_DIP, HandLandmark.WRIST
        ):
            state &= ~HandState.PINKY_UP

        self.state = state

    def draw_landmark(self, landmark, image, hand_state=None):
        if hand_state is None:
            color = (255, 0, 0)
        elif self.state & hand_state:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(
            image,
            (
                int(self.position[landmark.value].x * CAPTURE_WIDTH),
                int(self.position[landmark.value].y * CAPTURE_HEIGHT),
            ),
            5,
            color,
            3,
        )

    def get_position(self, a):
        return self.position[a.value]

    def get_state(self):
        return self.state


def calc_mouse_state(hand_state):
    if hand_state == HandState.OPEN:
        return MouseState.NORMAL
    elif hand_state == HandState.OPEN & ~HandState.THUMB_UP & ~HandState.INDEX_UP:
        return MouseState.LEFT
    elif hand_state == HandState.OPEN & ~HandState.MIDDLE_UP & ~HandState.RING_UP & ~HandState.PINKY_UP:
        return MouseState.RIGHT
    elif hand_state == HandState.OPEN & ~HandState.INDEX_UP & ~HandState.MIDDLE_UP & ~HandState.RING_UP:
        return MouseState.DOUBLE
    elif hand_state == HandState.CLOSED:
        return MouseState.SCROLL
    else:
        return MouseState.NONE


def update_window(cap_fps, is_moveable, hand, prev_time, frame, mouse_state):
    hand.draw_landmark(HandLandmark.WRIST, frame)
    hand.draw_landmark(HandLandmark.THUMB_TIP, frame, HandState.THUMB_UP)
    hand.draw_landmark(HandLandmark.INDEX_FINGER_TIP, frame, HandState.INDEX_UP)
    hand.draw_landmark(HandLandmark.MIDDLE_FINGER_TIP, frame, HandState.MIDDLE_UP)
    hand.draw_landmark(HandLandmark.RING_FINGER_TIP, frame, HandState.RING_UP)
    hand.draw_landmark(HandLandmark.PINKY_TIP, frame, HandState.PINKY_UP)

    cv2.putText(
        frame,
        f"cameraFPS: {cap_fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    process_fps = 1 / (time.perf_counter() - prev_time)
    cv2.putText(
        frame,
        f"processFPS: {process_fps:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )

    cv2.putText(frame, f"state: {mouse_state}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if is_moveable:
        cv2.rectangle(frame, (0, 0), (CAPTURE_WIDTH, CAPTURE_HEIGHT), (0, 0, 255), 5)

    cv2.imshow("Hand Detection", frame)


def main(args):
    # initialize
    cap = init_cap()
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    hand_detector = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    )

    global g_is_activate
    is_pre_detected = False

    pre_pos = Vector3(0, 0, 0)
    pre_state = MouseState.NONE

    hand = Hand(N_CONV)
    mouse = Controller()

    while cap.isOpened():
        prev_time = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            continue

        pos = Vector3(0, 0, 0)
        dx, dy = 0, 0
        hand_state = HandState.OPEN
        mouse_state = MouseState.NONE

        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        hand_detector_result = hand_detector.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if hand_detector_result.multi_hand_landmarks:
            assert len(hand_detector_result.multi_hand_landmarks) == 1

            hand_landmarks = hand_detector_result.multi_hand_landmarks[0]
            hand.update(hand_landmarks)

            hand_state = hand.get_state()
            mouse_state = calc_mouse_state(hand_state)

            pos = hand.get_position(HandLandmark.WRIST)

        is_detected = mouse_state != MouseState.NONE

        move_distance = calc_distance(pos, pre_pos)
        if move_distance > MOVE_THRESHOLD and is_pre_detected:
            dx = int((pos.x - pre_pos.x) * CAPTURE_WIDTH * SENSITIVITY)
            dy = int((pos.y - pre_pos.y) * CAPTURE_HEIGHT * SENSITIVITY)

        if g_is_activate:
            operate_mouse(pre_state, mouse, dx, dy, mouse_state)

        update_window(cap_fps, g_is_activate, hand, prev_time, frame, mouse_state)

        pre_pos = pos
        pre_state = mouse_state
        is_pre_detected = is_detected

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        if key == ord("z"):
            g_is_activate = not g_is_activate

    cap.release()
    cv2.destroyAllWindows()


def operate_mouse(pre_state, mouse, dx, dy, mouse_state):
    if mouse_state != pre_state:
        if pre_state == MouseState.LEFT:
            mouse.release(Button.left)
        elif pre_state == MouseState.RIGHT:
            mouse.release(Button.right)
        if mouse_state == MouseState.LEFT:
            mouse.press(Button.left)
        elif mouse_state == MouseState.RIGHT:
            mouse.press(Button.right)
        elif mouse_state == MouseState.DOUBLE:
            mouse.click(Button.left, 2)

    if mouse_state == MouseState.SCROLL:
        mouse.scroll(0, dy)
    elif mouse_state != MouseState.NONE:
        mouse.move(dx, dy)


def init_cap():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


if __name__ == "__main__":
    config = Config()
    main(config)
