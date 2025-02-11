from ultralytics import YOLO
import cv2
from moviepy import VideoFileClip
import os


video_path = "video.mp4"
# Blur ratio
blur_ratio = 50
# Boxes from [current_frame - frames_overlap:current_frame:frames_overlap+1] are blured in the current frame
frames_overlap = 1


# Load several YOLO model weights
model_names = [
"yolov10l-face.pt",
"yolov10m-face.pt",
"yolov11l-face.pt",
"yolov11n-face.pt",
"yolov11s-face.pt",
"yolov6m-face.pt",
"yolov8l-face.pt",
]
models = [YOLO(model_name) for model_name in model_names]

# Load the video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


boxes_list = []
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    boxes = None
    for model in models:
        results0 = model.predict(im0, show=False)
        boxes0 = results0[0].boxes.xyxy.cpu().tolist()
        if boxes is None:
            boxes = boxes0
        else:
            if boxes0 is not None:
                boxes.extend(boxes0)

    boxes_list.append(boxes)

cap.release()

# Now loop over and blur the frames and write them to a new temp video
video_writer = cv2.VideoWriter("object_blurring_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
print("Now blurring the detected objects in the video...")
frame_idx = 0
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    for boxes in boxes_list[max(0, frame_idx - frames_overlap):min(frame_idx + frames_overlap + 1, len(boxes_list))]:
        if boxes is not None:
            for box in boxes:
                # Coordinates
                xmax = int(box[0])
                ymin = int(box[1])
                xmin = int(box[2])
                ymax = int(box[3])

                # Extract the region of interest (ROI) corresponding to the detected face/object
                obj = im0[ymin:ymax, xmax:xmin]

                # Apply blur to the ROI
                blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                # Replace the original face with the blurred version
                im0[ymin:ymax, xmax:xmin] = blur_obj

    video_writer.write(im0)
    frame_idx += 1

cap.release()
video_writer.release()

# now get the audio from the original video and add it to the new video and save as an mp4
clip = VideoFileClip(video_path)
audio = clip.audio

final_clip = VideoFileClip("object_blurring_output.avi")
final_clip.audio = audio

split_video_path = video_path.split("/")
split_ = split_video_path[-1].split(".")[0]

output_file_name = split_ + "_blurred.mp4"
output_video_path = os.path.join("/".join(split_video_path[:-1]), output_file_name)
final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

# remove the temporary video file
os.remove("object_blurring_output.avi")