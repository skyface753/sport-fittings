
from src.models import AllLandmarks
from collections import defaultdict
from scipy.signal import find_peaks
import numpy as np
from src.models import Point


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Calculates the angle (in degrees) between three points, with b as the vertex."""
    # Create vectors from the vertex point b
    vec1 = a - b
    vec2 = c - b

    # Calculate dot product
    dot_product = vec1.x * vec2.x + vec1.y * vec2.y

    # Calculate magnitudes
    mag1 = np.sqrt(vec1.x**2 + vec1.y**2)
    mag2 = np.sqrt(vec2.x**2 + vec2.y**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0  # Avoid division by zero

    cosine_angle = dot_product / (mag1 * mag2)
    # Ensure cosine_angle is within [-1, 1] to prevent arccos errors due to floating point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def print_angle_stats(angles: list[float], angle_name: str):
    """Prints statistics for a list of angles."""
    if not angles:
        print(f"No {angle_name} angles calculated.")
        return
    print(f"\n--- {angle_name} Angle Statistics ---")
    print(f"Min angle: {min(angles):.2f}°")
    print(f"Max angle: {max(angles):.2f}°")
    print(f"Mean angle: {np.mean(angles):.2f}°")
    print(f"Median angle: {np.median(angles):.2f}°")
    print(f"Standard deviation: {np.std(angles):.2f}°")
    print("-" * (len(angle_name) + 22))


def calc_angles(all_frames_landmarks: AllLandmarks) -> dict[str, list[float]]:
    dynamic_angles = defaultdict(list)

    # right_wrist_series = all_frames_landmarks.get_landmark_series(
    #     "right_wrist")
    # right_elbow_series = all_frames_landmarks.get_landmark_series(
    #     "right_elbow")
    # right_shoulder_series = all_frames_landmarks.get_landmark_series(
    #     "right_shoulder")
    # right_hip_series = all_frames_landmarks.get_landmark_series("right_hip")
    # right_knee_series = all_frames_landmarks.get_landmark_series("right_knee")
    # right_ankle_series = all_frames_landmarks.get_landmark_series(
    #     "right_ankle")
    # right_heel_series = all_frames_landmarks.get_landmark_series("right_heel")
    # right_foot_index_series = all_frames_landmarks.get_landmark_series(
    #     "right_foot_index")

    # Calculate angles for each frame
    for i in range(len(all_frames_landmarks.frames_landmarks)):
        frame_lm = all_frames_landmarks.frames_landmarks[i]

        # Upper Body Angles
        if frame_lm.right_hip and frame_lm.right_shoulder and frame_lm.right_elbow:
            shoulder_angle = calculate_angle(
                frame_lm.right_hip, frame_lm.right_shoulder, frame_lm.right_elbow)
            dynamic_angles["Shoulder Angle (Hip-Shoulder-Elbow)"].append(
                shoulder_angle)

        if frame_lm.right_shoulder and frame_lm.right_elbow and frame_lm.right_wrist:
            elbow_angle = calculate_angle(
                frame_lm.right_shoulder, frame_lm.right_elbow, frame_lm.right_wrist)
            dynamic_angles["Elbow Angle (Shoulder-Elbow-Wrist)"].append(elbow_angle)

        # Lower Body Angles
        if frame_lm.right_hip and frame_lm.right_knee and frame_lm.right_ankle:
            knee_angle = calculate_angle(
                frame_lm.right_hip, frame_lm.right_knee, frame_lm.right_ankle)
            dynamic_angles["Knee Angle (Hip-Knee-Ankle)"].append(knee_angle)

        # Torso Angle (e.g., Shoulder-Hip-Horizontal)
        # This is a bit more complex as it requires a horizontal reference.
        # For simplicity, let's use the angle formed by hip, shoulder and a point horizontally from shoulder
        if frame_lm.right_hip and frame_lm.right_shoulder:
            # Create an artificial point horizontally from the shoulder
            horizontal_point = Point(
                frame_lm.right_hip.x + 100, frame_lm.right_hip.y)
            torso_angle = calculate_angle(
                frame_lm.right_shoulder, frame_lm.right_hip,  horizontal_point)
            dynamic_angles["Torso Angle (Shoulder-Hip-Horizontal)"].append(
                torso_angle)

        # Ankle Angle (Knee-Ankle-Foot Index) # NOT USED IN BIKE FITTING
        # if frame_lm.right_knee and frame_lm.right_ankle and frame_lm.right_foot_index:
        #     ankle_angle = calculate_angle(
        #         frame_lm.right_knee, frame_lm.right_ankle, frame_lm.right_foot_index)
        #     dynamic_angles["Ankle Angle (Knee-Ankle-Foot Index)"].append(ankle_angle)

    return dynamic_angles


def calc_knee_peaks(dynamic_angles: dict[str, list[float]]) -> tuple[list[int], np.ndarray]:
    # Calculate the Knee Angle (Hip-Knee-Ankle) peaks
    peak_detection_distance = 20
    knee_angles = np.array(dynamic_angles["Knee Angle (Hip-Knee-Ankle)"])
    # Adjust distance as needed
    peaks, _ = find_peaks(knee_angles, distance=peak_detection_distance)
    max_knee_angles = knee_angles[peaks]

    # drop all peaks, that are below the mean of the knee angles
    mean_knee_angle = np.mean(knee_angles)
    new_peaks = [peak for peak in peaks if knee_angles[peak] > mean_knee_angle]
    peaks = new_peaks
    max_knee_angles = knee_angles[peaks]
    return peaks, max_knee_angles


def angles_summary(dynamic_angles: dict[str, list[float]], avg_knee_angle_peaks: float):
    print("\n--- Cycling Angle Summary ---")

    # Calculate average angles (handle cases where angle list might be empty)
    avg_shoulder_angle = np.mean(dynamic_angles.get("Shoulder Angle (Hip-Shoulder-Elbow)", [
        0])) if dynamic_angles.get("Shoulder Angle (Hip-Shoulder-Elbow)") else 0
    avg_elbow_angle = np.mean(dynamic_angles.get("Elbow Angle (Shoulder-Elbow-Wrist)",
                                                 [0])) if dynamic_angles.get("Elbow Angle (Shoulder-Elbow-Wrist)") else 0
    avg_torso_angle = np.mean(dynamic_angles.get("Torso Angle (Shoulder-Hip-Horizontal)",
                                                 [0])) if dynamic_angles.get("Torso Angle (Shoulder-Hip-Horizontal)") else 0
    avg_knee_angle = np.mean(dynamic_angles.get("Knee Angle (Hip-Knee-Ankle)",
                                                [0])) if dynamic_angles.get("Knee Angle (Hip-Knee-Ankle)") else 0
    # avg_ankle_angle = np.mean(dynamic_angles.get("Ankle Angle (Knee-Ankle-Foot Index)",
    #                           [0])) if dynamic_angles.get("Ankle Angle (Knee-Ankle-Foot Index)") else 0

    # Get specific knee angles from identified frames
    # knee_angles_list = dynamic_angles.get("Knee Angle (Hip-Knee-Ankle)", [])
    # knee_angle_bottom = knee_angles_list[foot_bottom_index_mp] if 0 <= foot_bottom_index_mp < len(
    #     knee_angles_list) else float('nan')
    # knee_angle_top = knee_angles_list[foot_top_index_mp] if 0 <= foot_top_index_mp < len(
    #     knee_angles_list) else float('nan')

    # Avg of the knee angles peaks

    # print(f'Knee Angle (Bottom of Stroke): {knee_angle_bottom:.2f}°')
    # print(f'Knee Angle (Top of Stroke): {knee_angle_top:.2f}°')
    print(f'Knee Angle (Avg): {avg_knee_angle:.2f}°')
    print(f'Knee Angle Peaks (Avg): {avg_knee_angle_peaks:.2f}°')
    print(f'Shoulder Angle (Avg): {avg_shoulder_angle:.2f}°')
    print(f'Elbow Angle (Avg): {avg_elbow_angle:.2f}°')
    print(f'Torso Angle (Avg): {avg_torso_angle:.2f}°')
    # print(f'Ankle Angle (Avg): {avg_ankle_angle:.2f}°')
    print("-----------------------------\n")
