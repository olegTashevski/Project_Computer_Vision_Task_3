import cv2
import supervision as sv

from model import ModelProxy, YOLO_MODEL_TYPE
from utils import Helper
from main_constants import SOURCE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, SOURCE_VIDEO_PATH

# Function to detect and track objects
def process_video(video_path, output_folder):
    frame_count = 0
    image_count = 0

    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    model = ModelProxy(YOLO_MODEL_TYPE)

    frame_generator = sv.get_video_frames_generator(source_path=video_path)

    helper = Helper(SOURCE, video_info, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

    for frame in frame_generator:

        frame_count += 1
        if frame_count % 5 == 0:  # Process every 5th frame
            detections : sv.Detections = helper.filter_detections(model.infer(frame))  # Class IDs

            # Save results for CVAT
            annotations = []

            for i in range(len(detections.xyxy)):
                x_min, y_min, x_max, y_max = detections.xyxy[i]
                obj_id = detections.tracker_id[i]
                class_id = detections.class_id[i]

                annotation = {
                    "xmin": int(x_min),
                    "ymin": int(y_min),
                    "xmax": int(x_max),
                    "ymax": int(y_max),
                    "label": class_id,
                    "id": obj_id,
                }
                annotations.append(annotation)

            # Save frame & annotations
            image_name = f"frame_{image_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_folder + "/images", image_name), frame)
            save_cvat_format(annotations, image_name, output_folder)
            image_count += 1


import os
import xml.etree.ElementTree as ET


def save_cvat_format(annotations, image_name, output_folder, previous_tracks = {}):
    annotation_path = os.path.join(output_folder, "annotations.xml")

    # Extract frame number from image name (assumes "frame_0005.jpg" format)
    frame_number = int(image_name.split("_")[-1].split(".")[0])

    # Check if the annotation file already exists
    if os.path.exists(annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    else:
        # Create root element with metadata for interpolation mode
        root = ET.Element("annotations")

        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")

        ET.SubElement(task, "name").text = "Video_Annotation"
        ET.SubElement(task, "mode").text = "interpolation"
        ET.SubElement(task, "overlap").text = "0"
        ET.SubElement(task, "flipped").text = "False"
        ET.SubElement(task, "z_order").text = "False"

    # Track IDs present in this frame
    current_track_ids = {ann["id"] for ann in annotations}

    # Process detections in the current frame
    for ann in annotations:
        # Find or create a track with the same ID
        track = None
        for existing_track in root.findall("track"):
            if existing_track.attrib["id"] == str(ann["id"]):
                track = existing_track
                break

        if track is None:
            track = ET.SubElement(root, "track", id=str(ann["id"]), label=str(ann["label"]))

        # Add bounding box data for this frame
        ET.SubElement(track, "box",
                      frame=str(frame_number),
                      xtl=str(ann["xmin"]), ytl=str(ann["ymin"]),
                      xbr=str(ann["xmax"]), ybr=str(ann["ymax"]),
                      outside="0", occluded="0", keyframe="1")

    # Handle disappearing objects (mark "outside" in next frame)
    for track in root.findall("track"):
        track_id = int(track.attrib["id"])

        # If the track was in the previous frame but not in the current frame, mark it as outside
        if track_id in previous_tracks and track_id not in current_track_ids:
            ET.SubElement(track, "box",
                          frame=str(frame_number),
                          xtl=str(previous_tracks[track_id][0]),
                          ytl=str(previous_tracks[track_id][1]),
                          xbr=str(previous_tracks[track_id][2]),
                          ybr=str(previous_tracks[track_id][3]),
                          outside="1", occluded="0", keyframe="1")

    # Update previous track information
    previous_tracks.clear()
    for ann in annotations:
        previous_tracks[ann["id"]] = (ann["xmin"], ann["ymin"], ann["xmax"], ann["ymax"])

    tree = ET.ElementTree(root)
    tree.write(annotation_path)


# Main execution
def main():
    output_folder = "output_annotations"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_folder + "/images"):
        os.makedirs(output_folder + "/images")

    if not os.path.exists(output_folder + "/annotations"):
        os.makedirs(output_folder + "/annotations")
    # Process the video
    process_video(SOURCE_VIDEO_PATH, output_folder)


if __name__ == "__main__":
    main()
