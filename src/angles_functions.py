
from src.optimal_angles import ANGLE_SPECS
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

def calculate_seat_height_adjustment(
    current_angle: float,
    # optimal_range: tuple[float, float],
    next_optimal_angle: float,
    sensitivity_factor: float = 0.5 # Degrees per cm of seat height change (example value)
) -> float:
    """
    Calculates the required seat height adjustment (in cm) based on the current angle
    and an optimal range.

    Args:
        current_angle: The currently measured angle in degrees.
        optimal_range: A tuple (min_angle, max_angle) representing the desired
                       range for the angle in degrees.
        sensitivity_factor: How many degrees the angle changes for every 1 cm
                            of seat height adjustment. This value is CRITICAL
                            and needs to be determined empirically or via a
                            biomechanical model. A positive value means
                            increasing seat height increases the angle.

    Returns:
        The required seat height adjustment in centimeters.
        - Positive value: Increase seat height.
        - Negative value: Decrease seat height.
        - 0.0: Angle is within the optimal range.
    """
    # min_optimal_angle, max_optimal_angle = optimal_range
    print(f"Current angle: {current_angle:.2f}°")
    print(f"Next optimal angle: {next_optimal_angle:.2f}°")

    # if min_optimal_angle <= current_angle <= max_optimal_angle:
    #     return 0.0  # Angle is already within the optimal range

    if sensitivity_factor == 0:
        print("Warning: Sensitivity factor is zero, cannot calculate adjustment.")
        return 0.0

    angle_difference = next_optimal_angle - current_angle
    adjustment = angle_difference / sensitivity_factor
    return adjustment



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


def calc_angles(all_frames_landmarks: AllLandmarks, mode: str) -> dict[str, list[float]]:
    dynamic_angles = defaultdict(list)

    for frame_lm in all_frames_landmarks.frames_landmarks:
        for spec in ANGLE_SPECS:
            if mode not in spec.modes:
                continue  # skip angles not relevant for this mode

            p1, p2, p3 = spec.points
            if p3 == "horizontal_reference_point":
                p3 = Point(
                    getattr(frame_lm, p2).x + 100, getattr(frame_lm, p2).y)
                if all(hasattr(frame_lm, p) and getattr(frame_lm, p) for p in (p1, p2)):
                    angle = calculate_angle(
                        getattr(frame_lm, p1),
                        getattr(frame_lm, p2),
                        p3,
                    )
                    dynamic_angles[spec.label].append(angle)
            elif all(hasattr(frame_lm, p) and getattr(frame_lm, p) for p in (p1, p2, p3)):
                angle = calculate_angle(
                    getattr(frame_lm, p1),
                    getattr(frame_lm, p2),
                    getattr(frame_lm, p3),
                )
                dynamic_angles[spec.label].append(angle)
            else:
                print(f"Missing points for angle {spec.label} in frame. "
                      f"Points: {p1}, {p2}, {p3}")

    return dynamic_angles


def calc_peaks(dynamic_angles: dict[str, list[float]], angle_name="Knee Angle (Hip-Knee-Ankle)") -> tuple[list[int], np.ndarray, list[int], np.ndarray]:
    # Parameters
    peak_detection_distance = 20

    # Get the angle data
    angles = np.array(dynamic_angles[angle_name])

    # ------------------------
    # Find high peaks (maxima)
    peaks_high, _ = find_peaks(angles, distance=peak_detection_distance)
    mean_angle = np.mean(angles)
    peaks_high = [p for p in peaks_high if angles[p] > mean_angle]
    max_angles = angles[peaks_high]

    # ------------------------
    # Find low peaks (minima)
    peaks_low, _ = find_peaks(-angles, distance=peak_detection_distance)
    peaks_low = [p for p in peaks_low if angles[p] < mean_angle]
    min_angles = angles[peaks_low]

    return peaks_high, max_angles, peaks_low, min_angles


def angles_summary(dynamic_angles: dict[str, list[float]], avg_knee_angle_peaks_high: float,
                   avg_knee_angle_peaks_low: float):
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
    avg_hip_angle = np.mean(dynamic_angles.get("Hip Angle (Shoulder-Hip-Knee)",
                                               [0])) if dynamic_angles.get("Hip Angle (Shoulder-Hip-Knee)") else 0
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
    print(f'Knee Angle Peaks (High): {avg_knee_angle_peaks_high:.2f}°')
    print(f'Knee Angle Peaks (Low): {avg_knee_angle_peaks_low:.2f}°')
    print(f'Shoulder Angle (Avg): {avg_shoulder_angle:.2f}°')
    print(f'Elbow Angle (Avg): {avg_elbow_angle:.2f}°')
    print(f'Torso Angle (Avg): {avg_torso_angle:.2f}°')
    print(f'Hip Angle (Avg): {avg_hip_angle:.2f}°')
    # print(f'Ankle Angle (Avg): {avg_ankle_angle:.2f}°')
    print("-----------------------------\n")
