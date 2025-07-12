from src.models import AllLandmarks
import os
import cv2
from src.video_processing import process_video, process_video_with_running_mode
from src.visualisations import plot_image_with_points, show_top_and_bottom_knee_extension, show_video_with_landmarks, show_summarized_image
from src.angles_functions import calc_angles, calc_peaks, print_angle_stats, angles_summary
from src.kops import calculate_kops_and_get_frame_user_idea, KOPS_Point, print_kops_analysis_results
from src.visualisations import plot_angles_over_time
from src.recommondations import get_bike_fit_recommendations
import numpy as np
from src.recommondations import get_optimal_ranges
from src.pdf import create_summary_pdf
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
    from datetime import datetime
    curr_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"output/{fitting_mode}_{curr_date}"
    os.makedirs(output_dir, exist_ok=True)

    video_length, video_fps, video_width, video_height = get_video_stats(
        video_path)

    all_frames_landmarks, first_image_mp, first_frame_landmarks_mp = process_video_with_running_mode(
        video_path, video_length, VIDEO_DURATION, video_fps, video_width, video_height,
        output_path=os.path.join(
            output_dir, f"{fitting_mode}_mediapipe_output.mp4"),
        # full works better for cycling videos...
        model_path="pose_landmarker_full.task"
    )

    for frame_lm in all_frames_landmarks.frames_landmarks:
        frame_lm.from_gpu_to_cpu()

    # plot_image_with_points(
    #     first_image_mp, first_frame_landmarks_mp, with_foot_details=True)

    # inspect_first_frame_landmarks(first_frame_landmarks_mp)

    from src.optimal_angles import load_angle_specs_from_json
    angle_specs = load_angle_specs_from_json("angle_specs.json")

    # Calculate the angles
    dynamic_angles = calc_angles(
        all_frames_landmarks, fitting_mode, angle_specs)
    peaks_high, max_knee_angles, peaks_low, min_knee_angles = calc_peaks(
        dynamic_angles, angle_name="Knee Angle (Hip-Knee-Ankle)")
    _, _, hip_peaks_low, min_hip_angles = calc_peaks(
        dynamic_angles, angle_name="Hip Angle (Shoulder-Hip-Knee)")
    show_top_and_bottom_knee_extension(dynamic_angles, video_path, output_dir)
    show_video_with_landmarks(video_path, min(int(VIDEO_DURATION * video_fps), 200),  # dont go over the max index
                              all_frames_landmarks, output_dir, with_foot_details=True)

    show_video_with_landmarks(video_path, 45,  # dont go over the max index
                              all_frames_landmarks, output_dir, with_foot_details=True)
    # print(peaks_high)
    # get AllFramesLandmarks at the first peak index
    a = all_frames_landmarks.frames_landmarks[peaks_high[0]]
    print(a)
    # can be: heel, foot_index or ankle, ankle_vs_index
    foot_point_to_use = KOPS_Point.foot_index

    kops_analysis_results = calculate_kops_and_get_frame_user_idea(
        all_frames_landmarks, video_path, foot_point_to_use)

    print_kops_analysis_results(
        kops_analysis_results, foot_point_to_use, output_dir)

    angle_stats = []
    for angle_name, angles in dynamic_angles.items():
        angle_stats.append(print_angle_stats(angles, angle_name))
    with open(os.path.join(output_dir, "angle_stats.txt"), "w") as f:
        for line in angle_stats:
            if line is not None:
                f.write(" ".join(map(str, line)) + "\n")

    # write the whole dynamic angles to a file
    with open(os.path.join(output_dir, "dynamic_angles.txt"), "w") as f:
        for angle_name, angles in dynamic_angles.items():
            f.write(f"{angle_name}: {angles}\n")

    # TODO: remove this and replace with a visualization function
    show_summarized_image(
        first_image_mp, first_frame_landmarks_mp, output_dir)
    # cv2.imwrite(f"{fitting_mode}_summarized_image.png", summ_img)
    # cv2.imwrite(os.path.join(
    #     output_dir, f"{fitting_mode}_summarized_image.png"), summ_img)

    plot_angles_over_time(dynamic_angles, fitting_mode + "_angles_over_time",
                          angle_specs,
                          output_dir,
                          knee_peaks_high=peaks_high, max_knee_angles=max_knee_angles,
                          knee_peaks_low=peaks_low, min_knee_angles=min_knee_angles,
                          hip_peaks_low=hip_peaks_low, min_hip_angles=min_hip_angles,
                          mode=fitting_mode)

    avg_knee_angle_peaks_high, avg_knee_angle_peaks_low = calc_peaks_means(
        max_knee_angles, min_knee_angles)

    summary_output = angles_summary(dynamic_angles, avg_knee_angle_peaks_high,
                                    avg_knee_angle_peaks_low)
    for line in summary_output:
        print(line)
    # write to a file
    with open(f"{output_dir}/angles_summary.txt", "w") as f:
        for line in summary_output:
            f.write(line + "\n")

    recommendations = get_bike_fit_recommendations(
        dynamic_angles, avg_knee_angle_peaks_high, avg_knee_angle_peaks_low,
        kops_analysis_results["kops_value"],
        mode=fitting_mode, angle_specs=angle_specs)
    for i, rec in enumerate(recommendations):
        print(rec)
    # write recommendations to a file
    with open(f"{output_dir}/bike_fit_recommendations.txt", "w") as f:
        for i, rec in enumerate(recommendations):
            f.write(f"{rec}\n")

    # optimal_range_bottom = get_optimal_ranges(
    #     "Knee Angle (Hip-Knee-Ankle)", fitting_mode, angle_specs)[0]

    # print(f"Adjust seat height by: {calculate_seat_height_adjustment(current_angle=155, optimal_range=optimal_range_bottom, sensitivity_factor=4)} cm")
    create_summary_pdf(
        output_dir, f"{output_dir}/{fitting_mode}_summary.pdf",
        video_path.split("/")[-1], None)


if __name__ == "__main__":
    # TOP BAR
    # VIDEO_PATH = "datasets/cycling-sebastian/Hands_Top.mov"
    VIDEO_PATH = "datasets/cycling-sebastian/Hoods - Original v1.2.mov"

    main(VIDEO_PATH)

    # BOTTOM BAR
    # VIDEO_PATH = "datasets/cycling-sebastian/Hands_Bottom.mov"
    # main(VIDEO_PATH, fitting_mode="drop")

    # # AEROS
    # VIDEO_PATH = "datasets/cycling-sebastian/Aeros.mov"
    # main(VIDEO_PATH, fitting_mode="aero")
