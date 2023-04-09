import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


FOR_CENTER = [
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
]


SELECTED = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]
SELECTED_VALUES = [s.value for s in SELECTED]
SELECTED_KEY = {s.value: i for i, s in enumerate(SELECTED)}
center_map = [SELECTED_KEY[c] for c in FOR_CENTER]


def resize_img_by_h(img, desired_h):
    scale_factor = desired_h / img.shape[0]
    return cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)


def get_position(img, pose_landmarks, draw=True, selected=None):
    if selected is None:
        selected = SELECTED_VALUES
    if pose_landmarks is None:
        return []
    lm_list = []
    w_list = []
    for i, lm in enumerate(pose_landmarks.landmark):
        if i in selected:
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([cx, cy])
            w_list.append(lm.visibility)
            if draw:
                cv2.circle(
                    img,
                    (cx, cy),
                    int(max(img.shape[0], img.shape[1]) * 0.015),
                    (255, 0, 0),
                    cv2.FILLED,
                )
    if draw:
        mp_draw.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return np.array(lm_list), np.array(w_list)


def draw_position(img, pose_landmarks, lm_list):
    for i, (cx, cy) in enumerate(lm_list):
        cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
        cv2.putText(
            img,
            str(i),
            (cx, cy),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (127, 255, 127),
            1,
            cv2.LINE_AA,
        )
    if pose_landmarks is not None:
        mp_draw.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return img


def get_single_crop_param(keypoints, image):
    shift_ = (
        abs(
            max(
                keypoints[SELECTED_KEY[mp_pose.PoseLandmark.LEFT_SHOULDER]][1],
                keypoints[SELECTED_KEY[mp_pose.PoseLandmark.RIGHT_SHOULDER]][1],
            )
            - keypoints[SELECTED_KEY[mp_pose.PoseLandmark.NOSE]][1]
        )
        * 2
    )

    x_min = int(min(keypoints[:, 0]) - shift_)
    x_max = int(max(keypoints[:, 0]) + shift_)
    y_min = int(min(keypoints[:, 1]) - shift_)
    y_max = int(max(keypoints[:, 1]) + shift_)

    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0] - 1)
    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1] - 1)

    return y_min, y_max, x_min, x_max


def get_crop_param_image_to_desired(keypoints, image, desired_w, desired_h):
    shift_ = (
        abs(
            max(
                keypoints[SELECTED_KEY[mp_pose.PoseLandmark.LEFT_SHOULDER]][1],
                keypoints[SELECTED_KEY[mp_pose.PoseLandmark.RIGHT_SHOULDER]][1],
            )
            - keypoints[SELECTED_KEY[mp_pose.PoseLandmark.NOSE]][1]
        )
        * 2
    )

    x_min = int(min(keypoints[:, 0]) - shift_)
    x_max = int(max(keypoints[:, 0]) + shift_)
    y_min = int(min(keypoints[:, 1]) - shift_)
    y_max = int(max(keypoints[:, 1]) + shift_)

    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0] - 1)
    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1] - 1)

    h = y_max - y_min
    w = x_max - x_min

    w_adjust = int((h * desired_w / desired_h) - w)

    x_min -= w_adjust // 2
    x_max += w_adjust // 2

    y_min = max(y_min, 0)
    y_max = min(y_max, image.shape[0] - 1)
    x_min = max(x_min, 0)
    x_max = min(x_max, image.shape[1] - 1)

    return y_min, y_max, x_min, x_max


def overlay_transparent(img1, img2, opacity, t1=10, t2=255):
    # Resize img1 if needed
    if img1.shape[0] > img2.shape[0] or img1.shape[1] > img2.shape[1]:
        # Get aspect ratio of img1
        ratio = img1.shape[0] / img1.shape[1]
        # Resize img1 to fit img2
        new_width = img2.shape[1]
        new_height = int(new_width * ratio)
        img1 = cv2.resize(img1, (new_width, new_height))

    # Create a mask for img1
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img1_gray, t1, t2, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Resize mask to fit img2
    if mask.shape[0] > img2.shape[0] or mask.shape[1] > img2.shape[1]:
        mask = cv2.resize(mask, (img2.shape[1], img2.shape[0]))
        mask_inv = cv2.resize(mask_inv, (img2.shape[1], img2.shape[0]))

    # Apply mask to img1 and invert mask for img2
    img1_masked = cv2.bitwise_and(img1, img1, mask=mask)
    img2_masked = cv2.bitwise_and(img2, img2, mask=mask_inv)

    # Add masked images together with opacity
    result = cv2.addWeighted(img1_masked, opacity, img2_masked, 1 - opacity, 0)

    return result


def reshape_overlay(img_fg, img_bg, opacity, t1, t2):
    img_fg = resize_img_by_h(img_fg, img_bg.shape[0])

    left = img_bg.shape[1] - img_fg.shape[1]
    right = left // 2
    left = left - right
    img1_masked = cv2.copyMakeBorder(
        img_fg, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return overlay_transparent(img1_masked, img_bg, opacity, t1, t2)
