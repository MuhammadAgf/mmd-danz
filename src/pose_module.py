from multiprocessing import Queue, Process

import cv2
import numpy as np
import mediapipe as mp

import multiprocessing
import logging
import time

logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)


_SENTINEL_ = "_SENTINEL_"


def similarity(v1, v2):
    # Scale v1 and v2 to have the same size
    norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1_scaled = v1 * (norm_v2 / norm_v1)
    v2_scaled = v2 * (norm_v1 / norm_v2)

    # Center v1_scaled and v2_scaled to (0,0)
    center = np.mean(np.concatenate([v1_scaled, v2_scaled], axis=0), axis=0)
    v1_centered = v1_scaled - center
    v2_centered = v2_scaled - center

    # Calculate the cosine similarity between corresponding vectors in v1_centered and v2_centered
    dot_products = np.sum(v1_centered * v2_centered, axis=1, keepdims=True)
    norms = np.linalg.norm(v1_centered, axis=1, keepdims=True) * np.linalg.norm(v2_centered, axis=1, keepdims=True)
    similarity_scores = dot_products / norms

    # Calculate the average similarity score
    similarity = np.mean(similarity_scores)
    return similarity


def get_position(img, pose_landmarks, draw=True):
    if pose_landmarks is None:
        return []
    lm_list = []
    for i, lm in enumerate(pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lm_list.append([cx, cy])
    del lm_list[1:11]
    if draw:
        mp_draw.draw_landmarks(img, pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return np.array(lm_list)


def pose_process(
    in_queue: Queue,
    out_queue: Queue,
    log_queue: Queue,
    static_image_mode,
    model_complexity,
    min_detection_confidence,
    min_tracking_confidence,
):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while True:
        input_item = in_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break
        logger.info(str(type(input_item)))
        logger.info(str(len(input_item)))
        camera_frame, lm_vid, video_frame, counter = input_item
        if len(lm_vid) < 1:
            out_queue.put_nowait(np.nan)
            continue

        keypoints = pose.process(camera_frame)
        lm_cam = get_position(camera_frame, keypoints.pose_landmarks, False)
        score = similarity(lm_vid, lm_cam)
        out_queue.put_nowait(score)
        log_queue.put_nowait(
            (counter, camera_frame, video_frame, score)
        )

def log_process(log_queue: Queue):
    while True:

        input_item = log_queue.get(timeout=10)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        counter, camera_frame, video_frame, score = input_item
        cv2.imwrite(f'./logs/{counter}_camera.jpg', camera_frame)
        cv2.imwrite(f'./logs/{counter}_video.jpg', video_frame)
        with open('./logs/log.txt', 'a') as f:
            f.write(f'ct: {counter}, score: {score}\n')

class PoseProcess:

    def __init__(
            self,
            model_complexity,
            static_image_mode,
            min_detection_confidence,
            min_tracking_confidence,
        ):

        self._in_queue = Queue()
        self._out_queue = Queue()
        self._log_queue = Queue()

        self._pose_process = Process(target=pose_process, kwargs={
            "in_queue": self._in_queue,
            "out_queue": self._out_queue,
            'log_queue': self._log_queue,
            "static_image_mode": static_image_mode,
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        })
        self._pose_process.start()
        self._log_process = Process(target=log_process, kwargs={
            'log_queue': self._log_queue
        })
        self._log_process.start()
        self._counter = 0

    def get_pose(self):
        if not self._out_queue.empty():
            return self._out_queue.get(timeout=10)
        return None

    def infer_pose(self, camera_frame, lm_video, video_frame=None):
        self._counter += 1
        counter = self._counter
        self._in_queue.put_nowait(
            (camera_frame, lm_video, video_frame, counter)
        )

    def stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

        self._log_queue.put_nowait(_SENTINEL_)
        self._log_process.join(timeout=10)
