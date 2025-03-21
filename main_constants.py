
import numpy as np

SOURCE = np.array([(800, 410), (1125, 410), (1920, 850), (0, 850)])

TARGET_WIDTH = 32
TARGET_HEIGHT = 140

SOURCE_VIDEO_PATH = "m6_motorway_trim.mp4"
TARGET_VIDEO_PATH = "m6_motorway_trim_result.mp4"
IOU_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.5
SPEED_LIMIT = 100
YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=zFD1yUlct18"


TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)