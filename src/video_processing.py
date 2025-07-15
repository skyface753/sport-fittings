import os
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarkerResult, RunningMode
from mediapipe.tasks import python as mp_tasks
import cv2
import mediapipe as mp
from src.mediapipe import extract_landmarks_mediapipe_from_result
from src.models import AllLandmarks


# def process_video(video_path: str, video_length: int, video_duration: int, video_fps: int, video_width: int, video_height: int, output_path: str = None):

#     all_frames_landmarks = AllLandmarks()
#     all_frames_landmarks.clear()

#     mp_drawing = mp.solutions.drawing_utils
#     mp_pose = mp.solutions.pose

#     # Prepare video writer if output_path is provided
#     out = None
#     if output_path:
#         # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')

#         out = cv2.VideoWriter(output_path, fourcc,
#                               video_fps, (video_width, video_height))

#     with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75, model_complexity=2) as pose:

#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise TypeError(
#                 f"Error opening video stream or file for MediaPipe: {video_path}")

#         i = 0
#         first_image_mp = None
#         first_frame_landmarks_mp = None

#         j = 0

#         if video_duration <= 0 or video_duration is None:
#             video_duration = video_length

#         print("Processing video with MediaPipe...")
#         while cap.isOpened():
#             ret, image = cap.read()
#             if not ret:
#                 break
#             i += 1
#             CURR_PERCENTAGE = i/video_length*100
#             if i % 100 == 0:
#                 print(
#                     f"Processing frame {i}/{video_length}: ({CURR_PERCENTAGE:.2f}%)")
#             # Cut off the last 2 seconds of the video (if too long or irrelevant motion)
#             if i > video_length - 2*video_fps and video_length > 2*video_fps:
#                 break
#             if j > (video_fps * video_duration):
#                 break
#             j += 1  # counter for actually processed frames

#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image_rgb.flags.writeable = False

#             results = pose.process(image_rgb)

#             # Extract and store landmarks
#             frame_lm = extract_landmarks_mediapipe(
#                 results, video_width, video_height)
#             if frame_lm:
#                 all_frames_landmarks.append_frame(frame_lm)
#                 if first_image_mp is None:
#                     # Store the first image with landmarks drawn for visualization
#                     image_rgb.flags.writeable = True
#                     mp_drawing.draw_landmarks(
#                         image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#                     first_image_mp = image_rgb
#                     first_frame_landmarks_mp = frame_lm

#             # Draw landmarks on the frame and write to output video if needed
#             if output_path:
#                 image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#                 if results.pose_landmarks:
#                     mp_drawing.draw_landmarks(
#                         image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#                 out.write(image_bgr)

#         cap.release()
#         if out:
#             out.release()
#         print("MediaPipe processing finished.")
#         print(f"Processed {j} frames")
#         return all_frames_landmarks, first_image_mp, first_frame_landmarks_mp

# from mediapipe.tasks.python.vision.core.image import Image


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize_pose_landmarks(
    image,
    pose_result,
    video_width,
    video_height
) -> np.ndarray:
    """Draws pose landmarks on the input image and return it.
    Args:
      image: The input BGR image.
      pose_result: The PoseLandmarkerResult object.
      video_width: Width of the video frame.
      video_height: Height of the video frame.
    Returns:
      Image with pose landmarks drawn.
    """

    if pose_result.pose_landmarks:
        for pose_landmarks in pose_result.pose_landmarks:
            # Draw landmarks as circles
            for landmark in pose_landmarks:
                cx = int(landmark.x * video_width)
                cy = int(landmark.y * video_height)
                cv2.circle(image, (cx, cy), 5, TEXT_COLOR, -1)

    return image


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    # pose_landmarks_list = detection_result
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def process_video_with_running_mode(video_path: str, video_length: int, video_duration: int,
                                    video_fps: int, video_width: int, video_height: int,
                                    output_path: str = None, model_path: str = "pose_landmarker_full.task"):

    all_frames_landmarks = AllLandmarks()
    all_frames_landmarks.clear()

    # Prepare video writer if output_path is provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc,
                              video_fps, (video_width, video_height))

    # Load the landmarker with running mode VIDEO
    options = PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        min_pose_presence_confidence=0.5,
        output_segmentation_masks=False,
    )

    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TypeError(f"Error opening video stream or file: {video_path}")

    i = 0
    j = 0
    first_image = None
    first_frame_landmarks = None

    if video_duration <= 0 or video_duration is None:
        video_duration = video_length

    print("Processing video with PoseLandmarker (VIDEO mode)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        CURR_PERCENTAGE = i / video_length * 100
        if i % 100 == 0:
            print(
                f"Processing frame {i}/{video_length}: ({CURR_PERCENTAGE:.2f}%)")

        if i > video_length - 2 * video_fps and video_length > 2 * video_fps:
            break
        if j > (video_fps * video_duration):
            break
        j += 1

        # Convert to mediapipe Image format
        # mp_image = mp.Image(
        # image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int((i / video_fps) * 1000)

        result = landmarker.detect_for_video(
            mp_image, timestamp_ms)

        frame_landmarks = extract_landmarks_mediapipe_from_result(
            result, video_width, video_height)
        if frame_landmarks:
            all_frames_landmarks.append_frame(frame_landmarks)

            if first_image is None:
                annotated_image = frame.copy()
                for lm in result.pose_landmarks:
                    for point in lm:
                        cx = int(point.x * video_width)
                        cy = int(point.y * video_height)
                        cv2.circle(annotated_image, (cx, cy),
                                   3, (0, 255, 0), -1)
                first_image = annotated_image
                first_frame_landmarks = frame_landmarks

        # Draw landmarks if output video is requested
        if output_path:
            annotated_frame = frame.copy()
            # annotated_image = visualize_pose_landmarks(
            #     annotated_frame, result, video_width, video_height)
            annotated_image = draw_landmarks_on_image(annotated_frame, result)

            # for lm in result.pose_landmarks:
            #     for point in lm:
            #         cx = int(point.x * video_width)
            #         cy = int(point.y * video_height)
            #         cv2.circle(annotated_frame, (cx, cy), 2, (0, 255, 0), -1)
            out.write(annotated_image)

    cap.release()
    if out:
        out.release()
    landmarker.close()

    print("Processing finished.")
    print(f"Processed {j} frames")
    return all_frames_landmarks, first_image, first_frame_landmarks


def process_single_frame(frame, timestamp_ms, landmarker, image_width, image_height, all_frames_landmarks, out_writer: str = None):

    # Convert to mediapipe Image format
    # mp_image = mp.Image(
    # image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    result = landmarker.detect_for_video(
        mp_image, timestamp_ms)

    frame_landmarks = extract_landmarks_mediapipe_from_result(
        result, image_width, image_height)
    if frame_landmarks:
        all_frames_landmarks.append_frame(frame_landmarks)
    else:
        all_frames_landmarks.append_empty_frame()

    # Draw landmarks if output video is requested
    if out_writer:
        annotated_frame = frame.copy()
        # annotated_image = visualize_pose_landmarks(
        #     annotated_frame, result, video_width, video_height)
        annotated_image = draw_landmarks_on_image(annotated_frame, result)

        out_writer.write(annotated_image)

    return all_frames_landmarks, annotated_image


def get_video_stats(video_path):
    """
    Retrieves video statistics such as length, FPS, width, and height.
    """
    if not os.path.exists(video_path):
        raise TypeError("Video not found.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TypeError("Could not open video file.")

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # Release the cap after getting info

    return video_length, video_fps, video_width, video_height


def process_video_from_file(video_path: str, landmarker, output_dir: str, process_duration: int, fitting_mode: str = "hood"):

    video_length, video_fps, video_width, video_height = get_video_stats(
        video_path)
    all_frames_landmarks = AllLandmarks(video_width, video_height)
    all_frames_landmarks.clear()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise TypeError(f"Error opening video stream or file: {video_path}")

    i = 0
    j = 0
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = os.path.join(
        output_dir, f"{fitting_mode}_mediapipe_output.mp4")
    out = cv2.VideoWriter(output_path, fourcc,
                          video_fps, (video_width, video_height))

    if process_duration is None:
        process_duration = video_length

    all_video_frames = []

    print("Processing video with PoseLandmarker (VIDEO mode)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        CURR_PERCENTAGE = i / video_length * 100
        if i % 100 == 0:
            print(
                f"Processing frame {i}/{video_length}: ({CURR_PERCENTAGE:.2f}%)")

        if i > video_length - 2 * video_fps and video_length > 2 * video_fps:
            break
        if j > (video_fps * process_duration):
            break
        j += 1

        timestamp_ms = int((i / video_fps) * 1000)
        all_frames_landmarks, _ = process_single_frame(
            frame, timestamp_ms, landmarker, video_width, video_height, all_frames_landmarks, out_writer=out)
        all_video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    print("Processing finished.")
    print(f"Processed {j} frames")
    return all_frames_landmarks, all_video_frames, video_fps, video_length
