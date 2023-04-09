import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from tqdm import tqdm

from src.utils import FOR_CENTER, SELECTED, SELECTED_KEY, SELECTED_VALUES, center_map
from src.video_module import VideoProcessor

RESOLUTION = (1280, 720)

desired_w = RESOLUTION[0] * 2 / 5
desired_h = RESOLUTION[1]


def process(master_path, fps):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    video_path = master_path / "video.mp4"
    vid_proc = VideoProcessor(video_path=str(video_path))
    vid_proc.init_frame_dictionary(mp_pose=mp_pose)
    vid_proc.process_lm()
    vid_proc.get_all_frame_crop_param(desired_w, desired_h)

    with open(master_path / "frame_dictionary.pickle", "wb") as f:
        pickle.dump(vid_proc.frame_dictionary, f)

    X = [
        list((stat["lm_list"] / np.linalg.norm(stat["lm_list"])).ravel())
        if stat["lm_list"] is not None and len(stat["lm_list"]) > 0
        else list(np.zeros(26) - 100)
        for stat in vid_proc.frame_dictionary
    ]

    X = np.vstack(X)

    n_clusters = int(X.shape[0] / (fps / 2))
    km = MiniBatchKMeans(n_clusters=n_clusters)
    km.fit(X)

    km_labels = km.labels_

    X = [
        list((stat["lm_list"] / np.linalg.norm(stat["lm_list"])).ravel())
        if stat["lm_list"] is not None and len(stat["lm_list"]) > 0
        else list(np.zeros(26) - 100)
        for stat in vid_proc.frame_dictionary
    ]
    for i in range(len(X)):
        X[i] += [i / len(X)]

    db = DBSCAN(eps=0.1, min_samples=int(fps / 10), metric="manhattan").fit(X)
    db_labels = db.labels_
    label_df = pd.DataFrame({"km": km_labels, "dbs": db_labels})
    new_label = [(i, v) for i, v in zip(km_labels, db_labels)]
    res = pd.DataFrame({"labels": new_label})
    labels = new_label

    new_labels = {"original_label": [], "new_label": [], "start": [], "end": []}

    for label in set(labels):
        indices = list(sorted(res[res["labels"] == label].index))

        cur = indices[0]
        start = indices[0]
        end = indices[0]
        counter = 0

        for i in range(1, len(indices)):
            val = indices[i]
            if val != cur + 1:
                new_labels["original_label"].append(label)
                new_labels["new_label"].append((label, counter))
                new_labels["start"].append(start)
                new_labels["end"].append(end)

                cur = val
                start = val
                end = val

                counter += 1
            else:
                end = val
            cur = val

        new_labels["original_label"].append(label)
        new_labels["new_label"].append((label, counter))
        new_labels["start"].append(start)
        new_labels["end"].append(end)

    df = pd.DataFrame(new_labels)
    df["duration"] = df["end"] - df["start"] + 1

    sorted_df = df.sort_values("start").reset_index(drop=True)
    last = 0
    cues = []
    for duration in sorted_df["duration"].values:
        if duration + last >= fps - int(fps / 15):
            last = 0
            cues.append(True)
        else:
            cues.append(False)
            last += duration
    sorted_df["cues"] = cues
    cues = [
        dict(start=v["start"], end=v["end"], duration=v["duration"])
        for _, v in sorted_df[sorted_df["cues"]].iterrows()
    ]

    with open(master_path / "cues.pickle", "wb") as f:
        pickle.dump(cues, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a game using a cam feed")
    parser.add_argument("--fps", type=int, default=30, help="FPS of the video")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/rolling girl/",
        help="path to the desired data directory",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    fps = args.fps

    process(data_path, fps)
