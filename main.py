from src.optimal_angles import OptimalAngles
from src.models import AllLandmarks
import os
import cv2
from src.video_processing import process_video
from src.visualisations import plot_image_with_points, show_top_and_bottom_knee_extension, show_video_with_landmarks, show_summarized_image
from src.angles_functions import calc_angles, calc_knee_peaks, print_angle_stats, angles_summary
from src.kops import calculate_kops_and_get_frame_user_idea, KOPS_Point, print_kops_analysis_results
from src.visualisations import plot_angles_over_time
from src.recommondations import get_bike_fit_recommendations
import numpy as np


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


def main():
    VIDEO_PATH = "datasets/cycling-sebastian/Hands_Top.mov"
    VIDEO_DURATION = 2  # seconds

    # all_frames_landmarks = AllLandmarks()

    # Check if the Video exists
    if not os.path.exists(VIDEO_PATH):
        raise TypeError("Video not found.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise TypeError("Could not open video file.")

    VIDEO_LENGTH = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    VIDEO_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()  # Release the cap after getting info

    print(f"Video length: {VIDEO_LENGTH} frames")
    print(f"Video FPS: {VIDEO_FPS}")
    print(f"Video width: {VIDEO_WIDTH}")
    print(f"Video height: {VIDEO_HEIGHT}")

    all_frames_landmarks, first_image_mp, first_frame_landmarks_mp = process_video(
        VIDEO_PATH, VIDEO_LENGTH, VIDEO_DURATION, VIDEO_FPS, VIDEO_WIDTH, VIDEO_HEIGHT
    )

    for frame_lm in all_frames_landmarks.frames_landmarks:
        frame_lm.from_gpu_to_cpu()

    if first_image_mp is not None and first_frame_landmarks_mp is not None:
        plot_image_with_points(
            first_image_mp, first_frame_landmarks_mp, with_foot_details=True)
    else:
        print("No frames with detectable landmarks were processed by MediaPipe.")

    inspect_first_frame_landmarks(first_frame_landmarks_mp)

    # Calculate the angles

    dynamic_angles = calc_angles(all_frames_landmarks)
    peaks, max_knee_angles = calc_knee_peaks(dynamic_angles)
    show_top_and_bottom_knee_extension(dynamic_angles, VIDEO_PATH)
    show_video_with_landmarks(VIDEO_PATH, min(int(VIDEO_DURATION * VIDEO_FPS), 200),  # dont go over the max index
                              all_frames_landmarks, with_foot_details=True)

    # can be: heel, foot_index or ankle, ankle_vs_index
    foot_point_to_use = KOPS_Point.foot_index

    kops_analysis_results = calculate_kops_and_get_frame_user_idea(
        all_frames_landmarks, VIDEO_PATH, foot_point_to_use)

    print_kops_analysis_results(kops_analysis_results, KOPS_Point.foot_index)

    for angle_name, angles in dynamic_angles.items():
        print_angle_stats(angles, angle_name)

    # TODO: remove this and replace with a visualization function
    show_summarized_image(first_image_mp, first_frame_landmarks_mp)

    plot_angles_over_time(dynamic_angles, "Angles Over Time",
                          separate_figures=False, knee_peaks=peaks, max_knee_angles=max_knee_angles)

    # TODO: outsource this to a separate function
    avg_knee_angle_peaks = np.mean(
        max_knee_angles) if max_knee_angles.size > 0 else float('nan')

    angles_summary(dynamic_angles, avg_knee_angle_peaks)

    recommendations = get_bike_fit_recommendations(
        dynamic_angles, avg_knee_angle_peaks)
    for i, rec in enumerate(recommendations):
        print(rec)


if __name__ == "__main__":
    main()
