import gc

import cv2
import numpy as np
from tqdm import tqdm

from src.utils import (
    FOR_CENTER,
    SELECTED,
    SELECTED_KEY,
    SELECTED_VALUES,
    center_map,
    draw_position,
    get_crop_param_image_to_desired,
    get_position,
    get_single_crop_param,
)


def generator():
    while True:
        yield


def resize_to_desired_w(y_min, y_max, x_min, x_max, desired_w, desired_h, image):
    h = y_max - y_min
    w = x_max - x_min

    w_adjust = int((h * desired_w / desired_h) - w)

    if w_adjust < 0:
        h_adjust = int(((desired_h / desired_w) * w) - h)

        if h + h_adjust < image.shape[0]:
            y_max, y_min = shift_y_axes(y_max, y_min, h + h_adjust, image, idx=0)
        else:
            y_max, y_min = y_max, y_min - abs(h_adjust)
    else:
        if w + abs(w_adjust) < image.shape[1]:
            x_max, x_min = shift_y_axes(x_max, x_min, w + abs(w_adjust), image, 1)
        else:
            # TODO append black pixels
            raise NotImplemented()
    return (y_min, y_max, x_min, x_max)


def shift_y_axes(y_max, y_min, max_h, image, idx=0):
    # idx = 1 for x axes

    shift_h = max_h - (y_max - y_min)
    shift_h_up = int(shift_h / 2)
    shift_h_down = shift_h - shift_h_up

    y_min = max(y_min - shift_h_up, 0)
    y_max = min(y_max + shift_h_down, image.shape[idx] - 1)

    if max_h - (y_max - y_min) > 0:
        shift_h = max_h - (y_max - y_min)
        if y_max == image.shape[0] - 1:
            y_min = y_min - shift_h
        else:
            y_max = y_max + shift_h
        assert y_max - y_min == max_h
    return y_max, y_min


def preprocess_video(video_path, pose, preprocess_func=None):
    frame_dictionary = []

    video = cv2.VideoCapture(video_path)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(generator(), total=length):
        grabbed, frame = video.read()

        if not grabbed:
            break

        play_time = int(video.get(cv2.CAP_PROP_POS_MSEC))

        keypoints = pose.process(
            cv2.cvtColor(preprocess_func(frame), cv2.COLOR_BGR2RGB)
        )

        frame_dictionary.append(
            dict(play_time=play_time, pose_landmark=keypoints.pose_landmarks)
        )
    return frame_dictionary, length


class VideoProcessor:
    def __init__(self, video_path, frame_dictionary=None, mirror=False):
        self.video_path = video_path
        self.frame_dictionary = frame_dictionary
        self.video_cap = None
        self.mirror = mirror

        if self.mirror:
            self._preprocess_func = lambda x: cv.flip(x, 1)
        else:
            self._preprocess_func = lambda x: x

    def init_frame_dictionary(self, mp_pose=None, preprocess_func=None):
        if self.frame_dictionary is None:
            if preprocess_func is None:
                preprocess_func = self._preprocess_func

            pose = mp_pose.Pose()
            self.frame_dictionary, self.length = preprocess_video(
                self.video_path, pose, preprocess_func
            )
            del pose
            gc.collect()
        else:
            self.length = len(self.frame_dictionary)

    def init_play(self):
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.length = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def play_video(self):
        if self.video_cap is None:
            self.init_play()

        current_frame = 0
        while True:
            grabbed, frame = self.video_cap.read()
            if not grabbed:
                break
            yield current_frame + 1, self.frame_dictionary[current_frame], frame

    def get_frame(self, frame_no):
        if self.video_cap is None:
            self.init_play()

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
        grabbed, frame = self.video_cap.read()

        if self.mirror:
            frame = self._preprocess_func(frame)

        return self.frame_dictionary[frame_no], frame

    def get_msec(self, elapsed_time_ms):
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_time_ms)
        frame_no = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))

        grabbed, frame = self.video_cap.read()

        if self.mirror:
            frame = self._preprocess_func(frame)

        return self.frame_dictionary[frame_no], frame, frame_no

    def get_frame_no_from_msec(self, elapsed_time_ms):
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_time_ms)
        frame_no = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        return frame_no

    def process_lm(self, preprocess_func=None):
        if preprocess_func is None:
            preprocess_func = self._preprocess_func
        _, frame = self.get_frame(0)
        frame = preprocess_func(frame)

        for i, val in enumerate(self.frame_dictionary):
            if val["pose_landmark"] is None:
                lm_list, w_list = [], []
            else:
                lm_list, w_list = get_position(
                    frame, val["pose_landmark"], draw=False, selected=None
                )

            self.frame_dictionary[i]["lm_list"] = lm_list
            self.frame_dictionary[i]["w_list"] = w_list

    def process_image_crop(self, desired_w, desired_h, preprocess_func=None):
        # unstable (image glitchy)
        if preprocess_func is None:
            preprocess_func = self._preprocess_func

        crop_params = []

        first_crop_param = None
        last_crop_param = None

        stat, frame = self.get_frame(0)
        img = preprocess_func(frame)
        for i, v in tqdm(
            enumerate(self.frame_dictionary), total=len(self.frame_dictionary)
        ):
            if v["pose_landmark"] is not None:
                crop_param = get_crop_param_image_to_desired(
                    v["lm_list"], img, desired_w, desired_h
                )

                if first_crop_param is None:
                    first_crop_param = crop_param
                last_crop_param = crop_param
            else:
                crop_param = None
                if last_crop_param is not None:
                    crop_param = last_crop_param

            crop_params.append(crop_param)

        for i, v in tqdm(enumerate(crop_params), total=len(crop_params)):
            if v is None:
                crop_params[i] = first_crop_param

        for i, v in tqdm(enumerate(crop_params), total=len(crop_params)):
            self.frame_dictionary[i]["crop_params"] = v

    def get_all_frame_crop_param(self, desired_w, desired_h, preprocess_func=None):
        if preprocess_func is None:
            preprocess_func = self._preprocess_func

        crop_params = []

        first_crop_param = None
        last_crop_param = None

        stat, frame = self.get_frame(0)
        img = preprocess_func(frame)
        for i, v in tqdm(
            enumerate(self.frame_dictionary), total=len(self.frame_dictionary)
        ):
            if v["pose_landmark"] is not None:
                crop_param = get_single_crop_param(
                    v["lm_list"],
                    img,
                )

                if first_crop_param is None:
                    first_crop_param = crop_param
                last_crop_param = crop_param
            else:
                crop_param = None
                if last_crop_param is not None:
                    crop_param = last_crop_param

            crop_params.append(crop_param)

        for i, v in tqdm(enumerate(crop_params), total=len(crop_params)):
            if v is None:
                crop_params[i] = first_crop_param
        max_h, max_w = np.max(
            [
                (y_max - y_min, x_max - x_min)
                for y_min, y_max, x_min, x_max in crop_params
            ],
            axis=0,
        )

        for i, (y_min, y_max, x_min, x_max) in enumerate(crop_params):
            y_max, y_min = shift_y_axes(y_max, y_min, max_h, img, 0)
            x_max, x_min = shift_y_axes(x_max, x_min, max_w, img, 1)

            self.frame_dictionary[i]["crop_orig"] = y_min, y_max, x_min, x_max
            self.frame_dictionary[i]["crop_params"] = resize_to_desired_w(
                y_min, y_max, x_min, x_max, desired_w, desired_h, img
            )
