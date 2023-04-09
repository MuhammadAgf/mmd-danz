# MMD Danz Game

`mmd-danz` is a dancing game that requires players to mirror dance moves as they appear on the screen in real-time while watching themselves in a webcam preview. The game is written in Python and uses the `OpenCV` and `Mediapipe` libraries for image processing, and `ffpyplayer` for audio playback.

## Demo

Click the image below for the demo:

[![rolling girl demo](https://img.youtube.com/vi/6L7QownxoJg/0.jpg)](https://www.youtube.com/watch?v=6L7QownxoJg)


## Installation

To run `mmd-danz`, you need to have Python 3.x installed on your computer. Clone the project repository and navigate to the project directory:

```bash
git clone git@github.com:MuhammadAgf/mmd-danz.git
cd mmd-danz
```

Then, install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

To start the game, run the following command:

```bash
python main.py [--data-path DATA_PATH] [--cam-host CAM_URL]
```

The command-line arguments are:

* `--cam-host`: URL of the webcam feed or 0 for default camera.
* `--data-path`: path to the directory of the video file to be played.

For example, to start the game using a video file inside `data/rolling girl`, run the following command:

```bash
python main.py --data-path "./data/rolling girl/" --cam-host http://<URL>/camera/mjpeg
```

During the game, the player will see themselves in real-time in the webcam preview, and a series of dance moves that must be mirrored to earn points. The player's score, time elapsed, and current frame number are displayed in the top left corner of the screen.


## Data Path 
Link to the original video that is used as the sample above: [here](https://youtu.be/3JiWG0bRels).

for a data path to be valid as the directory to play the game, it needs 3 files inside it:

* `video.mp4` the video file to be played.
* `frame_dictionary.pickle`  previously generated pickle file with processed video (video.mp4) frames.
* `cues.pickle` previously generated pickle file with cues for the video.

use `preprocess.py` to generate the `frame_dictionary.pickle` and `cues.pickle`
```bash
python preprocess.py [--data-path DATA_PATH] [--fps FPS]
```

## Contributing

If you find a bug or would like to contribute to the project, please create a pull request or an issue on the [GitHub repository](https://github.com/MuhammadAgf/mmd-danz).
