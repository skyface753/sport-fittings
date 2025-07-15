from src.models import AllLandmarks
from src.models import Point
import cv2
from src.models import FrameLandmarks
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src.angles_functions import calculate_angle
from dataclasses import asdict
import os

# Maps display names to OptimalAngles attribute names
angle_label_to_attrs = {
    "Knee Angle (Hip-Knee-Ankle)": ["knee_extension_bottom", "knee_extension_top"],
    # "Knee Angle (Hip-Knee-Ankle)": "knee_extension_bottom",
    "Torso Angle (Shoulder-Hip-Horizontal)": ["torso_to_horizontal"],
    "Elbow Angle (Shoulder-Elbow-Wrist)": ["elbow_angle"],
    "Shoulder Angle (Hip-Shoulder-Elbow)": ["shoulder_angle"],
    "Hip Angle (Shoulder-Hip-Knee)": ["hip_angle"],
    # "Knee Angle Top Position": "knee_extension_top",
}


def plot_image_with_points(image, frame_landmarks: FrameLandmarks, output_dir: str, output_prefix="", with_foot_details=False, body_side="right"):
    """
    Plots an image with detected body landmarks for a specified side.

    Args:
        image: The image data to plot.
        frame_landmarks (FrameLandmarks): An object containing the detected landmarks.
        output_dir (str): The directory to save the output image.
        output_prefix (str, optional): A prefix for the output filename. Defaults to "".
        with_foot_details (bool, optional): Whether to plot detailed foot points. Defaults to False.
        body_side (str, optional): The side of the body to plot ("right" or "left"). Defaults to "right".
    """
    plt.figure(figsize=(16, 10))
    plt.imshow(image)

    marker_size = 5

    # Define common body parts and their colors/markers
    body_parts_data = {
        "wrist": ('r', 'o'),
        "elbow": ('g', 'o'),
        "shoulder": ('b', 'o'),
        "hip": ('c', 'o'),
        "knee": ('m', 'o'),
        "ankle": ('y', 'o'),
    }

    # Define connections for visual lines, now generic
    connections_template = [
        ("shoulder", "elbow"),
        ("elbow", "wrist"),
        ("shoulder", "hip"),
        ("hip", "knee"),
        ("knee", "ankle"),
    ]

    # Plot nose if it exists
    if frame_landmarks.nose:
        plt.plot(frame_landmarks.nose.x, frame_landmarks.nose.y, 'r', linestyle="None",
                 marker="o", markersize=marker_size, label="Nose")

    # Plot points for the specified side
    for part, (color, marker) in body_parts_data.items():
        point_attr_name = f"{body_side}_{part}"
        p = getattr(frame_landmarks, point_attr_name, None)
        if p:
            plt.plot(p.x, p.y, color=color, linestyle="None",
                     marker=marker, markersize=marker_size, label=f"{body_side.capitalize()} {part.replace('_', ' ').title()}")

    # Draw connections for the specified side
    for p1_part, p2_part in connections_template:
        p1_attr = f"{body_side}_{p1_part}"
        p2_attr = f"{body_side}_{p2_part}"
        p1 = getattr(frame_landmarks, p1_attr, None)
        p2 = getattr(frame_landmarks, p2_attr, None)
        if p1 and p2:
            plt.plot([p1.x, p2.x], [p1.y, p2.y],
                     color='lightgray', linewidth=1)

    if with_foot_details:
        foot_index_attr = f"{body_side}_foot_index"
        heel_attr = f"{body_side}_heel"

        foot_index_p = getattr(frame_landmarks, foot_index_attr, None)
        heel_p = getattr(frame_landmarks, heel_attr, None)

        if foot_index_p:
            plt.plot(foot_index_p.x, foot_index_p.y, 'darkblue',
                     linestyle="None", marker="o", markersize=marker_size, label=f"{body_side.capitalize()} Foot Index")
        if heel_p:
            plt.plot(heel_p.x, heel_p.y, 'darkgreen',
                     linestyle="None", marker="o", markersize=marker_size, label=f"{body_side.capitalize()} Heel")
        if foot_index_p and heel_p:
            plt.plot([foot_index_p.x, heel_p.x],
                     [foot_index_p.y, heel_p.y],
                     color='lightgray', linewidth=1)

    plt.legend()
    plt.grid(True)
    plt.title(f"Keypoints Detection - {body_side.capitalize()} Side")
    image_path = os.path.join(
        output_dir, f"{output_prefix}{body_side}_keypoints_detection.png")
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to free up memory


# def get_image_at_index(video_path: str, index: int):
#     """Retrieves a specific frame from the video."""
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print("Error opening video stream or file")
#         raise TypeError
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if index >= video_length or index < 0:
#         print(f'Invalid frame index: {index}. Video length: {video_length}')
#         cap.release()
#         return None

#     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
#     ret, frame = cap.read()
#     cap.release()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         return frame
#     return None


def save_frame_from_video(all_video_frames: list[np.ndarray], index: int, title: str, output_dir: str = "output_frames", output_prefix: str = ""):
    """
    Extracts a specific frame from the video, adds a title, saves it as an image file,
    and returns the path to the saved image.

    Args:
        video_path: Path to the video file.
        index: The index of the frame to extract.
        title: The title to add to the image.
        output_dir: The directory where the image will be saved. Defaults to "output_frames".

    Returns:
        The path to the saved image file, or None if the frame could not be retrieved.
    """
    # frame = get_image_at_index(
    #     video_path, index)  # You need to have this function defined
    frame = all_video_frames[index] if index < len(all_video_frames) else None

    if frame is not None:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Sanitize title for filename: replace spaces with underscores and remove special chars
        filename_title = "".join(
            c if c.isalnum() else "_" for c in title).strip("_")
        image_filename = f"{output_prefix}{filename_title}_frame_{index}.png"
        image_path = os.path.join(output_dir, image_filename)

        plt.figure(figsize=(16, 10))
        plt.imshow(frame)
        plt.title(title)
        plt.axis('off')

        plt.savefig(image_path, bbox_inches='tight',
                    pad_inches=0.1)  # Save the figure
        plt.close()  # Close the figure to free up memory

        print(f"Frame saved to: {image_path}")
        return image_path
    else:
        print(f"Could not retrieve frame at index {index} from the video.")
        return None


def add_text_to_image(image, text: str, pos: int, video_height: int, font_scale: float = 0.8, thickness: int = 1):
    """Adds text to an image at a specified position with adjustable font size and thickness."""
    height_offset = video_height / 20  # Adjust offset for better spacing
    # Use cv2.LINE_AA for anti-aliased text
    cv2.putText(image, text,
                (10, int(height_offset * pos)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  # Adjustable font scale
                (255, 0, 0),  # Red color
                thickness,  # Adjustable thickness
                cv2.LINE_AA)
    return image


def plot_angles_over_time(
    angles_data: dict[str, list[float]],
    title: str,
    angle_specs,
    output_dir: str,
    # optimal_ranges: dict[str, tuple[float, float]] | None = None,
    knee_peaks_high=None, max_knee_angles=None,
    knee_peaks_low=None, min_knee_angles=None,
    hip_peaks_low=None, min_hip_angles=None,
    # optimal_angles: OptimalAngles = OptimalAngles(),
    mode: str = "hood",
):
    """
    Plots angle values over time. Can plot all angles in subplots in one figure
    or each angle in a separate figure, with optional optimal range highlighting.
    Also adds peaks for "Knee Angle (Hip-Knee-Ankle)".

    Args:
        angles_data: A dictionary where keys are angle names (str) and values are lists of angle values (float).
        title_prefix: A string prefix for the plot titles (e.g., "MediaPipe" or "YOLOv8").
        separate_figures: If True, each angle will be plotted in its own subplot.
        save: If True, saves the figure(s) to a PNG file.
        optimal_ranges: Optional dict mapping angle names to (min, max) optimal range to highlight on plots.
    """
    if not angles_data:
        print(f"No angle data provided for {title} to plot.")
        return

    num_angles = len(angles_data)
    if num_angles == 0:
        print(f"No angle data provided for {title} to plot.")
        return

    # optimal_angles = OptimalAngles()
    # angles_dict = asdict(optimal_angles)

    # plt.figure(figsize=(15, 6))
    # fig, ax = plt.subplots(figsize=(15, 6))
    fig = plt.figure("AnglePlot", figsize=(10, 6))  # Named figure to reuse
    plt.clf()  # Clear the current figure
    ax = fig.add_subplot(111)

    # Corrected way to get a discrete colormap:
    # Get the 'tab10' colormap object
    cmap = plt.colormaps.get_cmap('tab10')
    # Create a ListedColormap with the desired number of colors from the 'tab10' colormap
    # This ensures we have distinct colors for each line up to num_angles
    # Use the first num_angles colors
    colors = ListedColormap(cmap.colors[:num_angles])

    # Track if knee angle peaks have been added to avoid duplicate legends
    knee_peaks_added = False
    knee_valleys_added = False
    hip_valleys_added = False

    for i, (angle_label, angles) in enumerate(angles_data.items()):
        if angles:
            line_color = colors(i)
            # Plot the angle series
            ax.plot(angles, label=angle_label, color=line_color)

            spec = next((s for s in angle_specs if s.label ==
                        angle_label and mode in s.modes), None)
            if spec:
                ranges = spec.optimal_ranges.get(mode, [])
                for (min_val, max_val) in ranges:
                    ax.axhspan(min_val, max_val, color=line_color, alpha=0.2)
            else:
                print(
                    f"No optimal ranges found for {angle_label} in the provided optimal angles. Maybe its normal (eg. no Torso Angle for Aeros).")

            # Add optimal range highlight
            # attr_names = angle_label_to_attrs.get(angle_name, [])
            # found = False
            # for attr_name in attr_names:
            #     if attr_name in angles_dict:
            #         min_val, max_val = angles_dict[attr_name]
            #         ax.axhspan(min_val, max_val,
            #                     color=line_color, alpha=0.2)
            #         found = True
            #         print(
            #             f"Highlighting optimal range for {angle_name}: {min_val}° to {max_val}°")
            # if not found:
            #     print(
            #         f"No optimal range found for {angle_name} in the provided optimal angles. Maybe its normal (eg. no Torso Angle for Aeros).")
            # else:
            #     print(
            #         f"No optimal range for {angle_name}. Skipping highlight.")

            # Add peaks for "Knee Angle (Hip-Knee-Ankle)"
            if angle_label == "Knee Angle (Hip-Knee-Ankle)":
                if knee_peaks_high is not None and max_knee_angles is not None:
                    ax.plot(knee_peaks_high, max_knee_angles, "x", color='red', markersize=8,
                            label='Knee Angle Peaks' if not knee_peaks_added else "_nolegend_")
                    knee_peaks_added = True  # Mark that peaks have been added
                if knee_peaks_low is not None and min_knee_angles is not None:
                    ax.plot(knee_peaks_low, min_knee_angles, "x", color='blue', markersize=8,
                            label='Knee Angle Valleys' if not knee_valleys_added else "_nolegend_")
                    knee_valleys_added = True  # Mark that valleys have been added
            if angle_label == "Hip Angle (Shoulder-Hip-Knee)":
                if hip_peaks_low is not None and min_hip_angles is not None:
                    ax.plot(hip_peaks_low, min_hip_angles, "x", color='green', markersize=8,
                            label='Hip Angle Valleys' if not hip_valleys_added else "_nolegend_")
                    hip_valleys_added = True

    ax.set_title(title)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True)
    ax.legend()  # Display legend for all lines and the peak marker
    fig.savefig(os.path.join(output_dir, f"{title}.png"),
                bbox_inches='tight', pad_inches=0.1)  # Save the figure
    # ax.close()  # Close the figure to free up memory
    # return fig
    fig.canvas.manager.set_window_title(title)  # Set the window title
    fig.tight_layout()
    fig.show()


def draw_angle_on_image_precise(image, p1: Point, p_vertex: Point, p2: Point, angle_value: float, color=(0, 255, 255), line_thickness=2, arc_radius=50, font_scale=0.7, font_thickness=2):
    """
    Alternative implementation that draws the angle arc using points along the arc.
    This gives more precise control over the arc direction.
    """
    if not all([p1, p_vertex, p2]):
        return image

    # Draw segments
    cv2.line(image, (int(p1.x), int(p1.y)),
             (int(p_vertex.x), int(p_vertex.y)), color, line_thickness)
    cv2.line(image, (int(p2.x), int(p2.y)),
             (int(p_vertex.x), int(p_vertex.y)), color, line_thickness)

    # Calculate vectors from vertex to the other points
    vec1 = np.array([p1.x - p_vertex.x, p1.y - p_vertex.y])
    vec2 = np.array([p2.x - p_vertex.x, p2.y - p_vertex.y])

    # Get the angle between vectors using atan2
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    # Ensure we draw the smaller arc
    angle_diff = angle2 - angle1
    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    # Create points along the arc
    num_points = max(int(abs(angle_diff) * 180 / np.pi / 5),
                     3)  # At least 3 points
    angles = np.linspace(angle1, angle1 + angle_diff, num_points)

    arc_points = []
    for angle in angles:
        x = int(p_vertex.x + arc_radius * np.cos(angle))
        y = int(p_vertex.y + arc_radius * np.sin(angle))
        arc_points.append((x, y))

    # Draw the arc using polylines
    arc_points = np.array(arc_points, dtype=np.int32)
    cv2.polylines(image, [arc_points], False, color,
                  line_thickness, cv2.LINE_AA)

    # Calculate bisector for text placement
    mid_angle = angle1 + angle_diff / 2
    text_offset_x = int(p_vertex.x + (arc_radius + 15) * np.cos(mid_angle))
    text_offset_y = int(p_vertex.y + (arc_radius + 15) * np.sin(mid_angle))

    # Add text with background for better readability
    text = f'{angle_value:.1f}'
    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

    # Draw text background rectangle
    cv2.rectangle(image,
                  (text_offset_x - 2, text_offset_y - text_size[1] - 2),
                  (text_offset_x + text_size[0] + 2, text_offset_y + 5),
                  (0, 0, 0), -1)  # Black background

    # Draw the text
    cv2.putText(image, text,
                (text_offset_x, text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, font_thickness, cv2.LINE_AA)

    return image


def show_summarized_image(image, frame_landmarks: FrameLandmarks, output_dir: str, body_side: str = "right"):
    """
    Shows a summarized image with average angles, specific knee angles, and bike fit recommendations,
    and draws the angles on the image for the specified body side.
    """
    summary_image = image.copy()

    # Define landmark attribute names dynamically based on body_side
    hip_lm_attr = f"{body_side}_hip"
    knee_lm_attr = f"{body_side}_knee"
    ankle_lm_attr = f"{body_side}_ankle"
    shoulder_lm_attr = f"{body_side}_shoulder"
    elbow_lm_attr = f"{body_side}_elbow"
    wrist_lm_attr = f"{body_side}_wrist"
    foot_index_lm_attr = f"{body_side}_foot_index"

    # Retrieve landmarks using getattr
    hip = getattr(frame_landmarks, hip_lm_attr, None)
    knee = getattr(frame_landmarks, knee_lm_attr, None)
    ankle = getattr(frame_landmarks, ankle_lm_attr, None)
    shoulder = getattr(frame_landmarks, shoulder_lm_attr, None)
    elbow = getattr(frame_landmarks, elbow_lm_attr, None)
    wrist = getattr(frame_landmarks, wrist_lm_attr, None)
    foot_index = getattr(frame_landmarks, foot_index_lm_attr, None)

    # --- Draw the angles on the image ---
    # Draw Knee Angle (Hip-Knee-Ankle)
    if hip and knee and ankle:
        knee_angle_val = calculate_angle(hip, knee, ankle)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    hip,
                                                    knee,
                                                    ankle,
                                                    knee_angle_val,
                                                    color=(0, 255, 0), arc_radius=80)  # Green

    # Draw Elbow Angle (Shoulder-Elbow-Wrist)
    if shoulder and elbow and wrist:
        elbow_angle_val = calculate_angle(shoulder, elbow, wrist)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    wrist,
                                                    elbow,
                                                    shoulder,
                                                    elbow_angle_val,
                                                    color=(255, 0, 255), arc_radius=60)  # Magenta

    # Draw Torso Angle (Shoulder-Hip-Horizontal)
    if hip and shoulder:
        # Define horizontal point relative to the hip for the current side
        # Assuming horizontal line extends positively (to the right of the image)
        horizontal_point = Point(hip.x + 100, hip.y)

        torso_angle_val = calculate_angle(shoulder, hip, horizontal_point)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    horizontal_point,
                                                    hip,
                                                    shoulder,
                                                    torso_angle_val,
                                                    color=(255, 255, 0), arc_radius=100)  # Yellow

    # Draw Ankle Angle (Knee-Ankle-Foot Index)
    if knee and ankle and foot_index:
        ankle_angle_val = calculate_angle(knee, ankle, foot_index)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    knee,
                                                    ankle,
                                                    foot_index,
                                                    ankle_angle_val,
                                                    color=(0, 255, 255), arc_radius=70)  # Cyan

    plt.figure(figsize=(16, 10))
    plt.imshow(summary_image)
    plt.title(f"Bike Fitting Summary - {body_side.capitalize()} Side")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"cycling_summary_{body_side}.png"),
                bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the plot to free up memory

    # Ensure the returned image is in BGR format if it's going to be used by OpenCV downstream
    image_bgr = cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR)
    return image_bgr

    cv2.imwrite(f'cycling_summary.png', image_bgr)
    print(f"Summary image saved as cycling_summary.png")


def show_top_and_bottom_knee_extension(dynamic_angles: dict[str, list[float]], all_video_frames, output_dir: str, output_prefix=""):
    # Find the frames for min/max knee extension
    knee_angles_list = dynamic_angles.get("Knee Angle (Hip-Knee-Ankle)", [])
    foot_bottom_index_mp = -1
    foot_top_index_mp = -1

    if knee_angles_list:
        # Max knee angle is usually at the bottom of the pedal stroke (most extended)
        foot_bottom_index_mp = np.argmax(knee_angles_list)
        # Min knee angle is usually at the top of the pedal stroke (most flexed)
        foot_top_index_mp = np.argmin(knee_angles_list)

        print(
            f"\nFoot at Bottom (Max Knee Extension) Frame Index (MediaPipe): {foot_bottom_index_mp}")
        print(
            f"Foot at Top (Min Knee Extension) Frame Index (MediaPipe): {foot_top_index_mp}")
        # show_video_at_index(video_path, foot_bottom_index_mp,
        #                     "Foot at Bottom (Max Knee Extension)")
        # show_video_at_index(video_path, foot_top_index_mp,
        #                     "Foot at Top (Min Knee Extension)")
        save_frame_from_video(all_video_frames, foot_bottom_index_mp,
                              "Foot at Bottom (Max Knee Extension)", output_dir=output_dir, output_prefix=output_prefix)
        save_frame_from_video(all_video_frames, foot_top_index_mp,
                              "Foot at Top (Min Knee Extension)", output_dir=output_dir, output_prefix=output_prefix)
    else:
        print("\nCould not calculate knee angles with MediaPipe to determine foot bottom/top frames.")


# get the video at a custom index, and apply the landmarks at that index


# def get_video_at_index(video_path, index: int, landmarks: AllLandmarks):
#     """Retrieves a specific frame from the video and applies landmarks."""
#     frame = get_image_at_index(video_path, index)
#     if frame is not None:
#         # Convert to RGB for plotting
#         #        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # Get the landmarks for this index
#         if 0 <= index < len(landmarks.frames_landmarks):
#             frame_landmarks = landmarks.frames_landmarks[index]
#             return frame, frame_landmarks
#     return None, None


def show_video_with_landmarks(all_video_frames, index: int, landmarks: AllLandmarks, output_dir, output_prefix="", with_foot_details=False, body_side="right"):
    """Displays a specific frame from the video with landmarks."""
    # frame_rgb, frame_landmarks = get_video_at_index(
    #     video_path, index, landmarks)
    frame_rgb = all_video_frames[index] if index < len(
        all_video_frames) else None
    frame_landmarks = landmarks.frames_landmarks[index] if index < len(
        landmarks.frames_landmarks) else None
    if frame_rgb is not None and frame_landmarks is not None:
        plot_image_with_points(frame_rgb, frame_landmarks,
                               output_dir, output_prefix, with_foot_details, body_side=body_side)
    else:
        print(f"Could not retrieve or display frame at index {index}")
