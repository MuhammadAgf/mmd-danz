import gc
import cv2
import numpy as np
from tqdm import tqdm


def generator():
    while True:
        yield


def preprocess_video(video_path, pose):

    frame_dictionary = []

    video = cv2.VideoCapture(video_path)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(generator(), total=length):
        grabbed, frame = video.read()

        if not grabbed:
            break

        play_time = int(video.get(cv2.CAP_PROP_POS_MSEC))

        keypoints = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frame_dictionary.append(
            dict(play_time=play_time, pose_landmark=keypoints.pose_landmarks)
        )
    return frame_dictionary, length


class VideoProcessor:
    def __init__(self, video_path, frame_dictionary=None, mp_pose=None):
        self.video_path = video_path
        self.frame_dictionary = frame_dictionary

        self.video_cap = None

        if self.frame_dictionary is None:
            pose = mp_pose.Pose()
            self.frame_dictionary, self.length = preprocess_video(video_path, pose)
            del pose
            gc.collect()
        else:
            self.length = len(frame_dictionary)

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

        return self.frame_dictionary[frame_no], frame

    def get_msec(self, elapsed_time_ms):
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, elapsed_time_ms)
        frame_no = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))

        grabbed, frame = self.video_cap.read()
        return self.frame_dictionary[frame_no], frame, frame_no
