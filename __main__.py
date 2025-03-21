from numpy.lib.function_base import average

from utils import Helper
import cv2
import supervision as sv
from speed_estimator import SpeedEstimator
from model import ModelProxy, YOLO_MODEL_TYPE, FASTER_RCNN_MODEL_TYPE, RETINA_NET_MODEL_TYPE
from main_constants import SOURCE_VIDEO_PATH, SOURCE, TARGET, CONFIDENCE_THRESHOLD, TARGET_VIDEO_PATH, IOU_THRESHOLD
import math

if __name__ == "__main__":

    model_types = {YOLO_MODEL_TYPE, FASTER_RCNN_MODEL_TYPE, RETINA_NET_MODEL_TYPE}

    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)

    speed_estimator = SpeedEstimator(video_info.fps, SOURCE, TARGET)

    helper = Helper(SOURCE, video_info, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    model_ratings = {}
    for model_type in model_types:
        model = ModelProxy(model_type)
        frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
        with sv.VideoSink(model_type + "_" + TARGET_VIDEO_PATH, video_info) as sink:
            frame_num = 0
            mean_true_positive = 0
            for frame in frame_generator:

                detections = helper.filter_detections(model.infer(frame))

                labels = speed_estimator.get_speed_estimates(detections)

                annotated_frame = helper.annotate_frame(frame, labels, detections)
                frame_num += 1
                if frame_num % 5 == 0:
                    T_P = helper.get_true_positive_rate(detections)
                    mean_true_positive = mean_true_positive + T_P
                sink.write_frame(annotated_frame)
                cv2.imshow("frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


            mean_true_positive = mean_true_positive / math.floor(frame_num / 5)
            model_ratings[model_type] = mean_true_positive
            cv2.destroyAllWindows()
        helper.reset()
        speed_estimator.reset()
    print(model_ratings)