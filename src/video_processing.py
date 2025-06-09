import cv2
import mediapipe as mp
from src.mediapipe import extract_landmarks_mediapipe
from src.models import AllLandmarks


def process_video(video_path: str, video_length: int, video_duration: int, video_fps: int, video_width: int, video_height: int):

    all_frames_landmarks = AllLandmarks()
    all_frames_landmarks.clear()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Use a context manager for MediaPipe Pose for proper resource release
    with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75, model_complexity=2) as pose:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise TypeError(
                f"Error opening video stream or file for MediaPipe: {video_path}")

        i = 0
        first_image_mp = None
        first_frame_landmarks_mp = None

        j = 0

        print("Processing video with MediaPipe...")
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            i += 1
            CURR_PERCENTAGE = i/video_length*100
            if i % 100 == 0:
                print(
                    f"Processing frame {i}/{video_length}: ({CURR_PERCENTAGE:.2f}%)")
            # Cut off the last 2 seconds of the video (if too long or irrelevant motion)
            if i > video_length - 2*video_fps and video_length > 2*video_fps:  # only apply if video is long enough
                break
            if j > (video_fps * video_duration):
                break
            j += 1  # counter for actually processed frames

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False  # To improve performance

            results = pose.process(image_rgb)

            # Extract and store landmarks
            frame_lm = extract_landmarks_mediapipe(
                results, video_width, video_height)
            if frame_lm:
                all_frames_landmarks.append_frame(frame_lm)
                if first_image_mp is None:
                    # Store the first image with landmarks drawn for visualization
                    image_rgb.flags.writeable = True
                    mp_drawing.draw_landmarks(
                        image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    first_image_mp = image_rgb
                    first_frame_landmarks_mp = frame_lm
            # else:
            #     print(f"No landmarks detected for frame {i}") # Optional: log when no landmarks are found

        cap.release()
        print("MediaPipe processing finished.")
        print(f"Processed {j} frames")
        return all_frames_landmarks, first_image_mp, first_frame_landmarks_mp
