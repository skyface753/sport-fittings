from src.models import FrameLandmarks, Point, MPLandmark


# def _extract_landmarks_mediapipe(results, image_width: int, image_height: int) -> FrameLandmarks | None:
#     """Extracts relevant landmarks from MediaPipe results into a FrameLandmarks object."""
#     if not results.pose_landmarks:
#         return None

#     frame_lm = FrameLandmarks()
#     pose_landmarks = results.pose_landmarks.landmark

#     # Helper to safely get landmark and convert to Point
#     def get_point(landmark_enum):
#         lm = pose_landmarks[landmark_enum.value]
#         # Only include if visible (confidence > threshold, MediaPipe uses visibility for this)
#         if lm.visibility > 0.7:  # You can adjust this threshold
#             return Point(lm.x * image_width, lm.y * image_height)
#         return None

#     frame_lm.nose = get_point(MPLandmark.NOSE)
#     frame_lm.right_wrist = get_point(MPLandmark.RIGHT_WRIST)
#     frame_lm.right_elbow = get_point(MPLandmark.RIGHT_ELBOW)
#     frame_lm.right_shoulder = get_point(MPLandmark.RIGHT_SHOULDER)
#     frame_lm.right_hip = get_point(MPLandmark.RIGHT_HIP)
#     frame_lm.right_knee = get_point(MPLandmark.RIGHT_KNEE)
#     frame_lm.right_ankle = get_point(MPLandmark.RIGHT_ANKLE)
#     frame_lm.right_foot_index = get_point(MPLandmark.RIGHT_FOOT_INDEX)
#     frame_lm.right_heel = get_point(MPLandmark.RIGHT_HEEL)

#     # Optional: Add left side if you plan to use it
#     # frame_lm.left_wrist = get_point(MPLandmark.LEFT_WRIST)
#     # frame_lm.left_elbow = get_point(MPLandmark.LEFT_ELBOW)
#     # frame_lm.left_shoulder = get_point(MPLandmark.LEFT_SHOULDER)
#     # frame_lm.left_hip = get_point(MPLandmark.LEFT_HIP)
#     # frame_lm.left_knee = get_point(MPLandmark.LEFT_KNEE)
#     # frame_lm.left_ankle = get_point(MPLandmark.LEFT_ANKLE)
#     # frame_lm.left_foot_index = get_point(MPLandmark.LEFT_FOOT_INDEX)
#     # frame_lm.left_heel = get_point(MPLandmark.LEFT_HEEL)

#     return frame_lm


def extract_landmarks_mediapipe_from_result(result, image_width: int, image_height: int) -> FrameLandmarks | None:
    """Extracts relevant landmarks from MediaPipe PoseLandmarkerResult into a FrameLandmarks object."""
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    frame_lm = FrameLandmarks()
    pose_landmarks = result.pose_landmarks[0]  # only one person expected

    # Helper to safely get landmark and convert to Point
    def get_point(landmark_enum):
        lm = pose_landmarks[landmark_enum.value]
        if hasattr(lm, "visibility") and lm.visibility < 0.7:
            return None
        return Point(lm.x * image_width, lm.y * image_height)

    frame_lm.nose = get_point(MPLandmark.NOSE)
    frame_lm.right_wrist = get_point(MPLandmark.RIGHT_WRIST)
    frame_lm.right_elbow = get_point(MPLandmark.RIGHT_ELBOW)
    frame_lm.right_shoulder = get_point(MPLandmark.RIGHT_SHOULDER)
    frame_lm.right_hip = get_point(MPLandmark.RIGHT_HIP)
    frame_lm.right_knee = get_point(MPLandmark.RIGHT_KNEE)
    frame_lm.right_ankle = get_point(MPLandmark.RIGHT_ANKLE)
    frame_lm.right_foot_index = get_point(MPLandmark.RIGHT_FOOT_INDEX)
    frame_lm.right_heel = get_point(MPLandmark.RIGHT_HEEL)
    frame_lm.right_index = get_point(
        MPLandmark.RIGHT_INDEX)  # Index finger tip
    frame_lm.right_pinky = get_point(
        MPLandmark.RIGHT_PINKY)  # Right pinky tip

    frame_lm.left_wrist = get_point(MPLandmark.LEFT_WRIST)
    frame_lm.left_elbow = get_point(MPLandmark.LEFT_ELBOW)
    frame_lm.left_shoulder = get_point(MPLandmark.LEFT_SHOULDER)
    frame_lm.left_hip = get_point(MPLandmark.LEFT_HIP)
    frame_lm.left_knee = get_point(MPLandmark.LEFT_KNEE)
    frame_lm.left_ankle = get_point(MPLandmark.LEFT_ANKLE)
    frame_lm.left_foot_index = get_point(MPLandmark.LEFT_FOOT_INDEX)
    frame_lm.left_heel = get_point(MPLandmark.LEFT_HEEL)
    frame_lm.left_index = get_point(MPLandmark.LEFT_INDEX)  # Index finger tip
    frame_lm.left_pinky = get_point(
        MPLandmark.LEFT_PINKY)  # Left pinky tip

    # Optional: Add left side if needed
    # frame_lm.left_wrist = get_point(MPLandmark.LEFT_WRIST)
    # ...

    return frame_lm
