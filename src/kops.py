
import matplotlib.pyplot as plt
from enum import Enum
from src.models import AllLandmarks, Point
from src.visualisations import get_image_at_index
import cv2


class KOPS_Point(Enum):
    heel = "heel"
    foot_index = "foot_index"
    ankle = "ankle"
    ankle_vs_index = "ankle_vs_index"


def calculate_kops_and_get_frame_user_idea(all_landmarks: AllLandmarks, video_path: str, foot_point_to_use: KOPS_Point) -> dict:
    """
    Calculates KOPS (Knee Over Pedal Spindle) for the right leg based on the user's idea:
    finds the frame where the right_foot_index has the highest X-coordinate (most right),
    and then calculates the horizontal distance between the knee and the ankle in that frame.

    Args:
        all_landmarks: An AllLandmarks object containing processed pose data.
        video_path: Path to the original video file.

    Returns:
        A dictionary containing:
        - 'kops_value': The KOPS horizontal distance for the identified frame.
        - 'identified_frame_index': Index of the frame where right_foot_index has max X.
        - 'best_kops_frame_image': The image of the identified frame with drawn landmarks and lines.
    """
    max_foot_x = -float('inf')
    identified_frame_index = -1

    # 1. Find the frame with the highest right_foot_index.x
    for i, frame_lm in enumerate(all_landmarks.frames_landmarks):
        if frame_lm.right_foot_index and frame_lm.right_foot_index.x > max_foot_x:
            max_foot_x = frame_lm.right_foot_index.x
            identified_frame_index = i

    kops_value = None
    best_kops_frame_image = None

    if identified_frame_index != -1:
        frame_lm_at_max_foot_x = all_landmarks.frames_landmarks[identified_frame_index]
        knee_point = frame_lm_at_max_foot_x.right_knee
        reference_point = None
        if foot_point_to_use == KOPS_Point.heel:
            reference_point = frame_lm_at_max_foot_x.right_heel
        elif foot_point_to_use == KOPS_Point.foot_index:
            reference_point = frame_lm_at_max_foot_x.right_foot_index
        elif foot_point_to_use == KOPS_Point.ankle:
            reference_point = frame_lm_at_max_foot_x.right_ankle
        elif foot_point_to_use == KOPS_Point.ankle_vs_index:
            a = frame_lm_at_max_foot_x.right_ankle
            b = frame_lm_at_max_foot_x.right_foot_index
            c = a + b
            reference_point = Point(c.x / 2, c.y / 2)

        if knee_point and reference_point:
            kops_value = reference_point.x - knee_point.x

            # Retrieve the image for visualization
            image = get_image_at_index(video_path, identified_frame_index)
            if image is not None:
                knee_x, knee_y = int(knee_point.x), int(knee_point.y)
                ankle_x, ankle_y = int(
                    reference_point.x), int(reference_point.y)
                # Green line for knee
                cv2.line(image, (knee_x, 0),
                         (knee_x, image.shape[0]), (0, 255, 0), 2)
                # Blue line for pedal spindle
                cv2.line(image, (ankle_x, 0),
                         (ankle_x, image.shape[0]), (255, 0, 0), 2)
                # Draw horizontal line connecting the two vertical lines at a mid-point
                line_y_level = min(knee_y, ankle_y) - 30
                if line_y_level < 0:
                    # Fallback if too high
                    line_y_level = max(knee_y, ankle_y) + 30
                # Yellow line for distance
                cv2.line(image, (knee_x, line_y_level),
                         (ankle_x, line_y_level), (0, 255, 255), 2)
                cv2.putText(image, f"KOPS: {kops_value:.2f} px", (min(knee_x, ankle_x), line_y_level - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                best_kops_frame_image = image

            # if image is not None:
            #     # Re-process with MediaPipe to get results for drawing (using static_image_mode for efficiency)
            #     with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose_static:
            #         # MediaPipe expects BGR, convert back if needed for drawing tools
            #         image_bgr_for_mp = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #         image_bgr_for_mp.flags.writeable = False
            #         results_static = pose_static.process(image_bgr_for_mp)

            #         if results_static.pose_landmarks:
            #             # , cv2.COLOR_RGB2BGR) # For drawing, ensure it's modifiable BGR
            #             image_to_draw = image_bgr_for_mp
            #             image_to_draw.flags.writeable = True

            #             # Draw landmarks
            #             mp_drawing.draw_landmarks(
            #                 image_to_draw, results_static.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            #             # Draw KOPS lines
            #             knee_x, knee_y = int(knee_point.x), int(knee_point.y)
            #             ankle_x, ankle_y = int(
            #                 reference_point.x), int(reference_point.y)

            #             # Draw vertical line from knee
            #             # Green line for knee
            #             cv2.line(image_to_draw, (knee_x, 0),
            #                      (knee_x, VIDEO_HEIGHT), (0, 255, 0), 2)
            #             # Draw vertical line from ankle (pedal spindle proxy)
            #             # Blue line for pedal spindle
            #             cv2.line(image_to_draw, (ankle_x, 0),
            #                      (ankle_x, VIDEO_HEIGHT), (255, 0, 0), 2)

            #             # Draw horizontal line connecting the two vertical lines at a mid-point
            #             # using a vertical level to ensure line is straight
            #             # Adjust y for visibility
            #             line_y_level = min(knee_y, ankle_y) - 30
            #             if line_y_level < 0:
            #                 # Fallback if too high
            #                 line_y_level = max(knee_y, ankle_y) + 30

            #             # Yellow line for distance
            #             cv2.line(image_to_draw, (knee_x, line_y_level),
            #                      (ankle_x, line_y_level), (0, 255, 255), 2)

            #             # Put text for KOPS value
            #             text_pos = (min(knee_x, ankle_x), line_y_level - 10)
            #             cv2.putText(image_to_draw, f"KOPS: {kops_value:.2f} px", text_pos,
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            #             # Convert back to RGB for matplotlib
            #             best_kops_frame_image = cv2.cvtColor(
            #                 image_to_draw, cv2.COLOR_BGR2RGB)

    results = {
        'kops_value': kops_value,
        'identified_frame_index': identified_frame_index,
        'best_kops_frame_image': best_kops_frame_image
    }

    return results


def print_kops_analysis_results(kops_analysis_results: dict, kops_point: KOPS_Point):
    print("\n--- KOPS Analysis (User's Idea) ---")
    kops_point_v = kops_point.value
    if kops_analysis_results['kops_value'] is not None:
        print(
            f"KOPS horizontal distance (at max right foot X): {kops_analysis_results['kops_value']:.2f} pixels")
        print(
            f"Negative value: Knee is forward of estimated pedal spindle ({kops_point_v}).")
        print(
            f"Positive value: Knee is behind estimated pedal spindle ({kops_point_v}).")

        if kops_analysis_results['identified_frame_index'] != -1:
            print(
                f"\nDisplaying frame where right foot ({kops_point_v}) is furthest to the right (Frame {kops_analysis_results['identified_frame_index']})")
            if kops_analysis_results['best_kops_frame_image'] is not None:
                plt.figure(figsize=(10, 8))
                plt.imshow(kops_analysis_results['best_kops_frame_image'])
                plt.title(
                    f"KOPS Visualization (Frame {kops_analysis_results['identified_frame_index']})")
                plt.axis('off')
                plt.show()
            else:
                print(
                    "Failed to retrieve or process the identified frame for visualization.")
    else:
        print("Could not calculate KOPS. Ensure right_foot_index, right_knee, and right_ankle landmarks are detected.")
