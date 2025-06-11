from src.models import AllLandmarks
import os
import cv2
from src.video_processing import process_video
from src.visualisations import plot_image_with_points, show_top_and_bottom_knee_extension, show_video_with_landmarks, show_summarized_image
from src.angles_functions import calc_angles, calc_peaks, print_angle_stats, angles_summary
from src.kops import calculate_kops_and_get_frame_user_idea, KOPS_Point, print_kops_analysis_results
from src.visualisations import plot_angles_over_time
from src.recommondations import get_bike_fit_recommendations
import numpy as np

# TODO: outsource this to a separate function


def calc_peaks_means(max_knee_angles, min_knee_angles):
    """
    Calculate the average of the max and min knee angles.
    Returns NaN if no peaks are found.
    """
    avg_knee_angle_peaks_high = np.mean(
        max_knee_angles) if max_knee_angles.size > 0 else float('nan')
    avg_knee_angle_peaks_low = np.mean(
        min_knee_angles) if min_knee_angles.size > 0 else float('nan')
    return avg_knee_angle_peaks_high, avg_knee_angle_peaks_low


def inspect_first_frame_landmarks(first_frame_landmarks_mp):
    print("\n--- Inspecting First Frame Landmarks (MediaPipe) for Missing Points ---")
    print(f"Nose: {first_frame_landmarks_mp.nose}")
    print(f"Right Shoulder: {first_frame_landmarks_mp.right_shoulder}")
    print(f"Right Elbow: {first_frame_landmarks_mp.right_elbow}")
    print(f"Right Wrist: {first_frame_landmarks_mp.right_wrist}")
    print(f"Right Hip: {first_frame_landmarks_mp.right_hip}")
    print(f"Right Knee: {first_frame_landmarks_mp.right_knee}")
    print(f"Right Ankle: {first_frame_landmarks_mp.right_ankle}")
    # Crucial for ankle angle
    print(f"Right Foot Index: {first_frame_landmarks_mp.right_foot_index}")
    print(f"Right Heel: {first_frame_landmarks_mp.right_heel}")
    print("----------------------------------------------------------------------")


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


def main(video_path, fitting_mode="hood"):
    VIDEO_DURATION = 5  # seconds

    # TOP BAR
    # video_path = "datasets/cycling-sebastian/Hands_Top.mov"
    # optimal_anlges = OptimalAngles()

    # BOTTOM BAR
    # video_path = "datasets/cycling-sebastian/Hands_Bottom.mov"
    # optimal_anlges = OptimalAnglesOnTheDrops()

    video_length, video_fps, video_width, video_height = get_video_stats(
        video_path)

    all_frames_landmarks, first_image_mp, first_frame_landmarks_mp = process_video(
        video_path, video_length, VIDEO_DURATION, video_fps, video_width, video_height
    )

    for frame_lm in all_frames_landmarks.frames_landmarks:
        frame_lm.from_gpu_to_cpu()

    plot_image_with_points(
        first_image_mp, first_frame_landmarks_mp, with_foot_details=True)

    inspect_first_frame_landmarks(first_frame_landmarks_mp)

    # Calculate the angles
    dynamic_angles = calc_angles(all_frames_landmarks, mode=fitting_mode)
    peaks_high, max_knee_angles, peaks_low, min_knee_angles = calc_peaks(
        dynamic_angles, angle_name="Knee Angle (Hip-Knee-Ankle)")
    _, _, hip_peaks_low, min_hip_angles = calc_peaks(
        dynamic_angles, angle_name="Hip Angle (Shoulder-Hip-Knee)")
    show_top_and_bottom_knee_extension(dynamic_angles, video_path)
    show_video_with_landmarks(video_path, min(int(VIDEO_DURATION * video_fps), 200),  # dont go over the max index
                              all_frames_landmarks, with_foot_details=True)

    # can be: heel, foot_index or ankle, ankle_vs_index
    foot_point_to_use = KOPS_Point.foot_index

    kops_analysis_results = calculate_kops_and_get_frame_user_idea(
        all_frames_landmarks, video_path, foot_point_to_use)

    print_kops_analysis_results(kops_analysis_results, KOPS_Point.foot_index)

    for angle_name, angles in dynamic_angles.items():
        print_angle_stats(angles, angle_name)

    # TODO: remove this and replace with a visualization function
    summ_img = show_summarized_image(first_image_mp, first_frame_landmarks_mp)
    cv2.imwrite(f"{fitting_mode}_summarized_image.png", summ_img)

    plot_angles_over_time(dynamic_angles, fitting_mode + "_angles_over_time",
                          knee_peaks_high=peaks_high, max_knee_angles=max_knee_angles,
                          knee_peaks_low=peaks_low, min_knee_angles=min_knee_angles,
                          hip_peaks_low=hip_peaks_low, min_hip_angles=min_hip_angles,
                          mode=fitting_mode)

    avg_knee_angle_peaks_high, avg_knee_angle_peaks_low = calc_peaks_means(
        max_knee_angles, min_knee_angles)

    angles_summary(dynamic_angles, avg_knee_angle_peaks_high,
                   avg_knee_angle_peaks_low)

    recommendations = get_bike_fit_recommendations(
        dynamic_angles, avg_knee_angle_peaks_high, avg_knee_angle_peaks_low, mode=fitting_mode)
    for i, rec in enumerate(recommendations):
        print(rec)
    # write recommendations to a file
    with open(f"{fitting_mode}_bike_fit_recommendations.txt", "w") as f:
        for i, rec in enumerate(recommendations):
            f.write(f"{i + 1}. {rec}\n")


if __name__ == "__main__":
    # TOP BAR
    VIDEO_PATH = "datasets/cycling-sebastian/Hands_Top.mov"
    main(VIDEO_PATH)

    # BOTTOM BAR
    VIDEO_PATH = "datasets/cycling-sebastian/Hands_Bottom.mov"
    main(VIDEO_PATH, fitting_mode="drop")

    # AEROS
    VIDEO_PATH = "datasets/cycling-sebastian/Aeros.mov"
    main(VIDEO_PATH, fitting_mode="aero")
