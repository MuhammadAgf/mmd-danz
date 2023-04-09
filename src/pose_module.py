import logging
import multiprocessing
import time
from multiprocessing import Process, Queue

import cv2
import mediapipe as mp
import numpy as np

from src.utils import center_map, draw_position, get_position

logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)


_SENTINEL_ = "_SENTINEL_"


def get_score_bin(v1, v2, w1, w2, center_map, thresh=0.925):
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)

    center = np.mean(norm_v1[center_map] - norm_v2[center_map], axis=0)

    center_v1 = norm_v1 - center
    center_v2 = norm_v2

    w = w1 * w2
    w[center_map] = np.nan

    score = np.nansum(
        w
        * ((1 - np.linalg.norm(center_v1 - center_v2, axis=1, keepdims=True)) > thresh)
        .astype(int)
        .ravel()
    ) / np.nansum(w)
    return score


def get_score(v1, v2, w1, w2, center_map):
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)

    center = np.mean(norm_v1[center_map] - norm_v2[center_map], axis=0)

    center_v1 = norm_v1 - center
    center_v2 = norm_v2

    w = w1 * w2
    w[center_map] = np.nan

    score = np.nansum(
        w * (1 - np.linalg.norm(center_v1 - center_v2, axis=1, keepdims=True)).ravel()
    ) / np.nansum(w)
    return score


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
        input_item = in_queue.get(timeout=20)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break
        logger.info(str(type(input_item)))
        logger.info(str(len(input_item)))
        camera_frame, lm_vid, w_vid, video_frame, counter = input_item
        if len(lm_vid) < 1:
            out_queue.put_nowait(np.nan)
            continue

        keypoints = pose.process(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
        # camera_frame = cv2.flip(camera_frame, 1)
        lm_cam, w_cam = get_position(camera_frame, keypoints.pose_landmarks, False)
        camera_frame = draw_position(camera_frame, keypoints.pose_landmarks, lm_cam)
        video_frame = draw_position(video_frame, None, lm_vid)
        score = get_score_bin(lm_vid, lm_cam, w_vid, w_cam, center_map)
        out_queue.put_nowait(score)
        log_queue.put_nowait((counter, camera_frame, video_frame, score))


def log_process(log_queue: Queue):
    while True:
        input_item = log_queue.get(timeout=20)
        if isinstance(input_item, type(_SENTINEL_)) and input_item == _SENTINEL_:
            break

        counter, camera_frame, video_frame, score = input_item
        cv2.imwrite(f"./logs/{counter}_camera.jpg", camera_frame)
        cv2.imwrite(f"./logs/{counter}_video.jpg", video_frame)
        with open("./logs/log.txt", "a") as f:
            f.write(f"ct: {counter}, score: {score}\n")


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

        self._pose_process = Process(
            target=pose_process,
            kwargs={
                "in_queue": self._in_queue,
                "out_queue": self._out_queue,
                "log_queue": self._log_queue,
                "static_image_mode": static_image_mode,
                "model_complexity": model_complexity,
                "min_detection_confidence": min_detection_confidence,
                "min_tracking_confidence": min_tracking_confidence,
            },
        )
        self._pose_process.start()
        self._log_process = Process(
            target=log_process, kwargs={"log_queue": self._log_queue}
        )
        self._log_process.start()
        self._counter = 0

    def get_pose(self):
        if not self._out_queue.empty():
            return self._out_queue.get(timeout=10)
        return None

    def infer_pose(self, camera_frame, lm_video, w_vid, video_frame=None):
        self._counter += 1
        counter = self._counter
        self._in_queue.put_nowait((camera_frame, lm_video, w_vid, video_frame, counter))

    def stop_pose_process(self):
        self._in_queue.put_nowait(_SENTINEL_)
        self._pose_process.join(timeout=10)

        self._log_queue.put_nowait(_SENTINEL_)
        self._log_process.join(timeout=10)
