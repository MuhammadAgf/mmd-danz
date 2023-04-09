import argparse
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np

# ffpyplayer for playing audio
from ffpyplayer.player import MediaPlayer

from src.fps_counter import CvFpsCalc
from src.image_util import add_text_to_image
from src.pose_module import PoseProcess
from src.utils import draw_position, reshape_overlay, resize_img_by_h
from src.video_module import VideoProcessor
from src.web_cam import WebCam

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

RESOLUTION = (1280, 720)

desired_w = RESOLUTION[0] * 2 / 5
desired_h = RESOLUTION[1]


def read_pickle(video_data_path):
    if video_data_path is None:
        return None
    with open(video_data_path, "rb") as f:
        x = pickle.load(f)
    return x


def join_camera_video(
    camera_frame,
    video_frame,
    video_stat,
    current_score,
    fps,
    time_elapsed,
    frame_no,
    flip=False,
    cue=None,
    cue_stat=None,
    cue_frame=None,
    last_cue_end=None,
):
    if video_stat["pose_landmark"] is not None:
        video_frame = draw_position(
            img=video_frame,
            pose_landmarks=video_stat["pose_landmark"],
            lm_list=video_stat["lm_list"],
        )

    y_min, y_max, x_min, x_max = cue_stat["crop_orig"]
    cue_frame = cue_frame[y_min:y_max, x_min:x_max]

    y_min, y_max, x_min, x_max = video_stat["crop_params"]

    if y_min < 0:
        video_frame = cv2.copyMakeBorder(
            video_frame[0:y_max, x_min:x_max],
            abs(y_min),
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
    else:
        video_frame = video_frame[y_min:y_max, x_min:x_max]

    video_frame = resize_img_by_h(video_frame, desired_h)
    camera_frame = resize_img_by_h(camera_frame, desired_h)
    if flip:
        video_frame = cv2.flip(video_frame, 1)
        cue_frame = cv2.flip(cue_frame, 1)

    deno = cue["end"] - last_cue_end + 1
    alpha = (
        max(0, min(frame_no - last_cue_end + 1, cue["end"] - last_cue_end + 1))
        / deno
        * 0.5
    )
    camera_frame = reshape_overlay(cue_frame, camera_frame, alpha, 25, 25)
    result_image = cv2.hconcat([video_frame, camera_frame])

    result_image = add_text_to_image(
        result_image,
        f"\n scores: {current_score}\n fps: {fps}\n time elapsed: {time_elapsed}\n frame: {frame_no}, cue: {cue['end']}, {cue['start']}",
        top_left_xy=(1, 1),
        font_color_rgb=(255, 255, 255),
        # bg_color_rgb=(255, 255, 255),
        font_thickness=1,
    )
    return result_image


def get_readable_time(millis):
    seconds = (millis / 1000) % 60
    seconds = int(seconds)
    minutes = (millis / (1000 * 60)) % 60
    minutes = int(minutes)
    hours = (millis / (1000 * 60 * 60)) % 24

    res = "%d:%d:%d" % (hours, minutes, seconds)
    return res


def play_game(
    video_path, realtime_camera_capture, video_data_path=None, cues_path=None
):
    print("prepare realtime pose detector")
    pose_realtime = PoseProcess(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0,
    )

    print("prepare video processor")
    vid_proc = VideoProcessor(
        video_path,
        frame_dictionary=read_pickle(video_data_path),
    )
    vid_proc.init_frame_dictionary()

    cur_frame = -1
    max_frame = vid_proc.length

    print("prepare cues")
    cues = read_pickle(cues_path)
    cue = None
    cue_frame = None
    last_cue_end = 0
    if cues is not None:
        cue = cues.pop(0)
        cue_stat, cue_frame = vid_proc.get_frame(cue["end"])

    print("capture first camera")
    camera_frame = realtime_camera_capture.get()
    pose_realtime.infer_pose(camera_frame, [], [])
    time.sleep(1)
    result = pose_realtime.get_pose()
    print("Result", result)

    print("Start Media Player")
    media_player = MediaPlayer(video_path)

    start_time = time.time()

    fps_ct = CvFpsCalc(buffer_len=10)
    video_stat, video_frame = vid_proc.get_frame(0)

    scores = []
    values = []
    last_score_calculate = 0
    while cur_frame < max_frame:
        _, _ = media_player.get_frame()
        # Check if any similarity values are ready
        res = pose_realtime.get_pose()
        if res is not None:
            scores.append(res)

        elapsed = (time.time() - start_time) * 1000  # msec

        camera_frame = realtime_camera_capture.get()
        video_stat, video_frame, cur_frame = vid_proc.get_msec(elapsed)

        lm_video = video_stat["lm_list"]
        w_vid = video_stat["w_list"]
        pose_landmark_vid = video_stat["pose_landmark"]

        elapsed = (time.time() - start_time) * 1000  # msec
        if elapsed >= last_score_calculate + 1000 or cue is not None:
            if cue is None or cur_frame >= cue["end"]:
                print(cur_frame, cue)
                pose_realtime.infer_pose(camera_frame, lm_video, w_vid, video_frame)
                last_score_calculate = (time.time() - start_time) * 1000

                if cue is not None:
                    if len(cues) > 0:
                        last_cue_end = cue["end"]
                        cue = cues.pop(0)
                        cue_stat, cue_frame = vid_proc.get_frame(cue["end"])

        play_time = video_stat["play_time"]

        last_score = np.nan if len(scores) == 0 else scores[-1]
        processed_frame = join_camera_video(
            cv2.flip(camera_frame, 1),
            video_frame,
            video_stat,
            f"last: {np.nanmean([last_score]):0.4f}\n avg: {np.nanmean(scores):0.4f}\n sum: {np.nansum(scores):0.4f}",
            fps_ct.get(),
            get_readable_time(play_time),
            cur_frame,
            True,
            cue,
            cue_stat,
            cue_frame,
            last_cue_end,
        )

        cv2.imshow("Video", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        cur_frame += 1

    pose_realtime.stop_pose_process()
    vid_proc.video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Play a game using a cam feed')
    parser.add_argument('--cam-host', type=str, default='0', help='URL of the webcam feed or 0 for default camera')
    parser.add_argument('--data-path', type=str, default='./data/lukaluka/', help='path to the desired data directory')

    args = parser.parse_args()

    cam_host = args.cam_host
    if cam_host.isdigit():
        cam_host = int(cam_host)

    data_path = args.data_path
    video_path = data_path + "video.mp4"
    video_data_path = data_path + "frame_dictionary.pickle"
    cues_path = data_path + "cues.pickle"

    play_game(video_path, WebCam(cam_host, True), video_data_path, cues_path)
