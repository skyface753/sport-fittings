
import matplotlib.pyplot as plt
from enum import Enum
from src.models import AllLandmarks, Point
# from src.visualisations import get_image_at_index
import cv2


class KOPS_Point(Enum):
    heel = "heel"
    foot_index = "foot_index"
    ankle = "ankle"
    # ankle_vs_index = "ankle_vs_index"
    # heel_3_4_index = "heel_3_4_index"  # 3/4 of the way from heel to foot_index


def calculate_kops_and_get_frame_user_idea(all_landmarks: AllLandmarks, all_video_frames, foot_point_to_use: KOPS_Point,
                                           body_side: str = "right") -> dict:
    """
    Calculates KOPS (Knee Over Pedal Spindle) for the right leg based on the user's idea:
    finds the frame where the right_foot_index has the highest X-coordinate (most right),
    and then calculates the horizontal distance between the knee and the ankle in that frame.

    Args:
        all_landmarks: An AllLandmarks object containing processed pose data.
        video_path: Path to the original video file.

    Returns:
        A dictionary containing:
        - 'kops_value': The KOPS horizontal distance for the identified frame. (Positive means knee is behind the pedal spindle, negative means knee is forward.)
        - 'identified_frame_index': Index of the frame where right_foot_index has max X.
        - 'best_kops_frame_image': The image of the identified frame with drawn landmarks and lines.
    """
    max_foot_x = -float('inf')
    identified_frame_index = -1

    # 1. Find the frame with the highest right_foot_index.x
    for i, frame_lm in enumerate(all_landmarks.frames_landmarks):
        foot_index = getattr(frame_lm, f"{body_side}_foot_index")
        if foot_index:
            if body_side == "left":
                if foot_index.x < max_foot_x or identified_frame_index == -1:
                    max_foot_x = foot_index.x
                    identified_frame_index = i
            else:
                if foot_index.x > max_foot_x:
                    max_foot_x = foot_index.x
                    identified_frame_index = i

    kops_value = None
    best_kops_frame_image = None

    if identified_frame_index != -1:
        frame_lm_at_max_foot_x = all_landmarks.frames_landmarks[identified_frame_index]
        # knee_point = frame_lm_at_max_foot_x.right_knee
        knee_point = getattr(frame_lm_at_max_foot_x, f"{body_side}_knee")
        attribute_name = f"{body_side}_{foot_point_to_use.value}"

        # Get the reference point using getattr
        reference_point = getattr(frame_lm_at_max_foot_x, attribute_name)

        # reference_point = None
        # if foot_point_to_use == KOPS_Point.heel:
        #     reference_point = frame_lm_at_max_foot_x.right_heel
        # elif foot_point_to_use == KOPS_Point.foot_index:
        #     reference_point = frame_lm_at_max_foot_x.right_foot_index
        # elif foot_point_to_use == KOPS_Point.ankle:
        #     reference_point = frame_lm_at_max_foot_x.right_ankle
        # elif foot_point_to_use == KOPS_Point.ankle_vs_index:
        #     a = frame_lm_at_max_foot_x.right_ankle
        #     b = frame_lm_at_max_foot_x.right_foot_index
        #     c = a + b
        #     reference_point = Point(c.x / 2, c.y / 2)
        # elif foot_point_to_use == KOPS_Point.heel_3_4_index:
        #     a = frame_lm_at_max_foot_x.right_heel
        #     b = frame_lm_at_max_foot_x.right_foot_index
        #     c = a + b
        #     reference_point = Point(c.x * 0.75, c.y * 0.75)

        if knee_point and reference_point:
            kops_value = reference_point.x - knee_point.x

            # Retrieve the image for visualization
            # image = get_image_at_index(video_path, identified_frame_index)
            image = all_video_frames[identified_frame_index] if identified_frame_index < len(
                all_video_frames) else None
            if image is not None:
                knee_x, knee_y = int(knee_point.x), int(knee_point.y)
                ankle_x, ankle_y = int(
                    reference_point.x), int(reference_point.y)
                # Green line for knee
                cv2.line(image, (knee_x, 0),
                         (knee_x, image.shape[0]), (0, 255, 0), 2)  # RGB: (0, 255, 0) for green
                # Red line for pedal spindle
                cv2.line(image, (ankle_x, 0),
                         (ankle_x, image.shape[0]), (255, 0, 0), 2)  # RGB: (255, 0, 0) for red
                # Draw horizontal line connecting the two vertical lines at a mid-point
                line_y_level = min(knee_y, ankle_y) - 30
                if line_y_level < 0:
                    # Fallback if too high
                    line_y_level = max(knee_y, ankle_y) + 30
                # Blue line for distance
                cv2.line(image, (knee_x, line_y_level),
                         (ankle_x, line_y_level), (0, 0, 255), 2)  # RGB: (0, 0, 255) for blue
                # Dynamically adjust font scale based on image height
                font_scale = max(0.7, image.shape[0] / 700.0)
                legend_font_scale = max(0.5, image.shape[0] / 1400.0)

                cv2.putText(
                    image,
                    f"KOPS: {kops_value:.2f} px",
                    (min(knee_x, ankle_x), line_y_level - int(10 * font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                # Add legend, e.g., "Green: Knee, Blue: Pedal Spindle, Yellow: KOPS Distance"
                cv2.putText(
                    image,
                    "Green: Knee, Red: Pedal Spindle, Blue: KOPS Distance",
                    (10, image.shape[0] - int(10 * legend_font_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    legend_font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                # Add colored points for knee and reference point
                cv2.circle(image, (knee_x, knee_y), 5,
                           (0, 255, 0), -1)  # Green for knee
                # Red for reference point
                cv2.circle(image, (ankle_x, ankle_y), 5, (255, 0, 0), -1)
                # Save the image for output
                best_kops_frame_image = image
    results = {
        'kops_value': kops_value,
        'identified_frame_index': identified_frame_index,
        'best_kops_frame_image': best_kops_frame_image
    }

    return results


def print_kops_analysis_results(
    kops_analysis_results: dict,
    kops_point: KOPS_Point,
    output_dir: str,
    output_prefix: str = "",
    body_side: str = "right"
):
    print(
        f"\n--- KOPS Analysis (User's Idea, {body_side.capitalize()} Side) ---")
    kops_point_v = kops_point.value
    if kops_analysis_results['kops_value'] is not None:
        print(
            f"KOPS horizontal distance (at max {body_side} foot X): {kops_analysis_results['kops_value']:.2f} pixels")
        print(
            f"Negative value: Knee is forward of estimated pedal spindle ({kops_point_v}).")
        print(
            f"Positive value: Knee is behind estimated pedal spindle ({kops_point_v}).")

        if kops_analysis_results['identified_frame_index'] != -1:
            print(
                f"\nDisplaying frame where {body_side} foot ({kops_point_v}) is furthest to the {body_side} (Frame {kops_analysis_results['identified_frame_index']})")
            if kops_analysis_results['best_kops_frame_image'] is not None:
                img = kops_analysis_results['best_kops_frame_image']
                # Convert
                # CV2 uses BGR format
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{output_dir}/{output_prefix}kops_visualization_{body_side}_{kops_point_v}.png",
                            img)
            else:
                print(
                    "Failed to retrieve or process the identified frame for visualization.")
    else:
        print(
            f"Could not calculate KOPS. Ensure {body_side}_foot_index, {body_side}_knee, and {body_side}_ankle landmarks are detected.")
