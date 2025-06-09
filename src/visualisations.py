from src.models import AllLandmarks
from src.models import Point
import cv2
from src.models import FrameLandmarks
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from src.angles_functions import calculate_angle
from src.optimal_angles import OptimalAngles
from dataclasses import asdict

# Maps display names to OptimalAngles attribute names
angle_label_to_attr = {
    "Knee Angle (Hip-Knee-Ankle)": "knee_extension_bottom",
    "Torso Angle (Shoulder-Hip-Horizontal)": "torso_to_horizontal",
    "Elbow Angle (Shoulder-Elbow-Wrist)": "elbow_angle",
    "Shoulder Angle (Hip-Shoulder-Elbow)": "shoulder_angle",
    "Knee Angle Top Position": "knee_extension_top",
}


def plot_image_with_points(image, frame_landmarks: FrameLandmarks, with_foot_details=False):
    """Plots an image with detected body landmarks."""
    plt.figure(figsize=(16, 10))  # to set the figure size
    plt.imshow(image)

    marker_size = 5  # Increased marker size for better visibility

    # List of tuples: (landmark_point_attribute, color, marker_style)
    # Define connections for visual lines
    connections = [
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("right_shoulder", "right_hip"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    points_to_plot = [
        (frame_landmarks.nose, 'r', 'o', "Nose"),
        (frame_landmarks.right_wrist, 'r', 'o', "Right Wrist"),
        (frame_landmarks.right_elbow, 'g', 'o', "Right Elbow"),
        (frame_landmarks.right_shoulder, 'b', 'o', "Right Shoulder"),
        (frame_landmarks.right_hip, 'c', 'o', "Right Hip"),
        (frame_landmarks.right_knee, 'm', 'o', "Right Knee"),
        (frame_landmarks.right_ankle, 'y', 'o', "Right Ankle"),
    ]

    for p, color, marker, label in points_to_plot:
        if p:
            plt.plot(p.x, p.y, color=color, linestyle="None",
                     marker=marker, markersize=marker_size, label=label)

    # Draw connections
    for p1_attr, p2_attr in connections:
        p1 = getattr(frame_landmarks, p1_attr, None)
        p2 = getattr(frame_landmarks, p2_attr, None)
        if p1 and p2:
            plt.plot([p1.x, p2.x], [p1.y, p2.y],
                     color='lightgray', linewidth=1)

    if with_foot_details:
        if frame_landmarks.right_foot_index:
            plt.plot(frame_landmarks.right_foot_index.x, frame_landmarks.right_foot_index.y, 'darkblue',
                     linestyle="None", marker="o", markersize=marker_size, label="Right Foot Index")
        if frame_landmarks.right_heel:
            plt.plot(frame_landmarks.right_heel.x, frame_landmarks.right_heel.y, 'darkgreen',
                     linestyle="None", marker="o", markersize=marker_size, label="Right Heel")
        if frame_landmarks.right_foot_index and frame_landmarks.right_heel:
            plt.plot([frame_landmarks.right_foot_index.x, frame_landmarks.right_heel.x],
                     [frame_landmarks.right_foot_index.y,
                         frame_landmarks.right_heel.y],
                     color='lightgray', linewidth=1)

    plt.legend()
    plt.grid(True)
    plt.title("Keypoints Detection")
    plt.show()


def get_image_at_index(video_path: str, index: int):
    """Retrieves a specific frame from the video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
        raise TypeError
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if index >= video_length or index < 0:
        print(f'Invalid frame index: {index}. Video length: {video_length}')
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def show_video_at_index(video_path: str, index: int, title: str):
    """Displays a specific frame from the video with a title."""
    frame = get_image_at_index(video_path, index)
    if frame is not None:
        plt.figure(figsize=(16, 10))  # to set the figure size
        plt.imshow(frame)
        plt.title(title)
        plt.axis('off')  # Hide axes for cleaner image
        plt.show()
    else:
        print(f"Could not retrieve frame at index {index}")


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
    title_prefix: str,
    separate_figures: bool = False,
    save: bool = True,
    # optimal_ranges: dict[str, tuple[float, float]] | None = None,
    knee_peaks=None,
    max_knee_angles=None,
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
        print(f"No angle data provided for {title_prefix} to plot.")
        return

    num_angles = len(angles_data)
    if num_angles == 0:
        print(f"No angle data provided for {title_prefix} to plot.")
        return

    optimal_angles = OptimalAngles()
    angles_dict = asdict(optimal_angles)

    if separate_figures:
        # Dynamically determine rows and columns for subplots
        rows = int(np.ceil(num_angles / 2)) if num_angles > 0 else 1
        cols = 2 if num_angles > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        # Ensure axes is always iterable, even for a single subplot
        axes = axes.flatten() if num_angles > 1 else [axes]

        for i, (angle_name, angles) in enumerate(angles_data.items()):
            if not angles:
                print(f"No data for {angle_name}. Skipping plot.")
                continue

            ax = axes[i]
            ax.plot(angles)
            ax.set_title(f"{angle_name}")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Angle (°)")
            ax.grid(True)

            # Draw optimal range if available
            if angles_dict and angle_name in angles_dict:
                min_val, max_val = angles_dict[angle_name]
                ax.axhspan(min_val, max_val, color='green',
                           alpha=0.2, label='Optimal Range')
                ax.legend()

            # Add peaks specifically for "Knee Angle (Hip-Knee-Ankle)"
            if angle_name == "Knee Angle (Hip-Knee-Ankle)":
                if knee_peaks is not None and max_knee_angles is not None:
                    # knee_angles = np.array(angles)
                    # peaks, _ = find_peaks(knee_angles, distance=peak_detection_distance)
                    # Plot peaks as 'x' markers
                    ax.plot(knee_peaks, max_knee_angles, "x",
                            color='red', markersize=8, label='Peaks')
                    ax.legend()

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"{title_prefix} Angles Over Time", fontsize=16)
        # Adjust layout to prevent title overlap
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save:
            plt.savefig(f'{title_prefix}.png')
        plt.show()
    else:  # Plot all angles in one figure
        plt.figure(figsize=(15, 6))

        # Corrected way to get a discrete colormap:
        # Get the 'tab10' colormap object
        cmap = plt.colormaps.get_cmap('tab10')
        # Create a ListedColormap with the desired number of colors from the 'tab10' colormap
        # This ensures we have distinct colors for each line up to num_angles
        # Use the first num_angles colors
        colors = ListedColormap(cmap.colors[:num_angles])

        # Track if knee angle peaks have been added to avoid duplicate legends
        knee_peaks_added = False

        for i, (angle_name, angles) in enumerate(angles_data.items()):
            if angles:
                line_color = colors(i)
                # Plot the angle series
                plt.plot(angles, label=angle_name, color=line_color)

                # TODO: MERGE WITH THE LEGEND ABOVE
                # Add optimal range highlight
                attr_name = angle_label_to_attr.get(angle_name)
                if attr_name and attr_name in angles_dict:
                    min_val, max_val = angles_dict[attr_name]
                    plt.axhspan(min_val, max_val, color=line_color, alpha=0.2)
                # if angles_dict and angle_name in angles_dict:
                #     min_val, max_val = angles_dict[angle_name]
                #     # Use the same color as the line for the optimal range highlight
                #     plt.axhspan(min_val, max_val, color=line_color, alpha=0.2)
                else:
                    print(
                        f"No optimal range for {angle_name}. Skipping highlight.")

                # Add peaks for "Knee Angle (Hip-Knee-Ankle)"
                if angle_name == "Knee Angle (Hip-Knee-Ankle)":
                    if knee_peaks is not None and max_knee_angles is not None:
                        plt.plot(knee_peaks, max_knee_angles, "x", color='red', markersize=8,
                                 label='Knee Angle Peaks' if not knee_peaks_added else "_nolegend_")
                        knee_peaks_added = True  # Mark that peaks have been added
                    # knee_angles = np.array(angles)
                    # peaks, _ = find_peaks(knee_angles, distance=peak_detection_distance)
                    # # Plot peaks as 'x' markers. Add label only once.
                    # plt.plot(peaks, knee_angles[peaks], "x", color='red', markersize=8, label='Knee Angle Peaks' if not knee_peaks_added else "_nolegend_")
                    # knee_peaks_added = True # Mark that peaks have been added

        plt.title(title_prefix)
        plt.xlabel("Frame Number")
        plt.ylabel("Angle (degrees)")
        plt.grid(True)
        plt.legend()  # Display legend for all lines and the peak marker
        if save:
            plt.savefig(f'{title_prefix}.png')
        plt.show()


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


def show_summarized_image(image, frame_landmarks: FrameLandmarks):
    """
    Shows a summarized image with average angles, specific knee angles, and bike fit recommendations,
    and draws the angles on the image.
    """
    summary_image = image.copy()

    # knee_angles_list = angles_data.get("Knee Angle (Hip-Knee-Ankle)", [])
    # knee_angle_bottom = knee_angles_list[bottom_idx] if 0 <= bottom_idx < len(
    #     knee_angles_list) else 0
    # knee_angle_top = knee_angles_list[top_idx] if 0 <= top_idx < len(
    #     knee_angles_list) else 0

    # avg_shoulder_angle = np.mean(angles_data.get("Shoulder Angle (Hip-Shoulder-Elbow)", [
    #                              0])) if angles_data.get("Shoulder Angle (Hip-Shoulder-Elbow)") else 0
    # avg_elbow_angle = np.mean(angles_data.get("Elbow Angle (Shoulder-Elbow-Wrist)", [
    #                           0])) if angles_data.get("Elbow Angle (Shoulder-Elbow-Wrist)") else 0
    # avg_torso_angle = np.mean(angles_data.get("Torso Angle (Shoulder-Hip-Horizontal)", [
    #                           0])) if angles_data.get("Torso Angle (Shoulder-Hip-Horizontal)") else 0
    # avg_knee_angle = np.mean(angles_data.get(
    #     "Knee Angle (Hip-Knee-Ankle)", [0])) if angles_data.get("Knee Angle (Hip-Knee-Ankle)") else 0
    # avg_ankle_angle = np.mean(angles_data.get("Ankle Angle (Knee-Ankle-Foot Index)", [
    #                           0])) if angles_data.get("Ankle Angle (Knee-Ankle-Foot Index)") else 0

    # text_font_scale = 0.6
    # text_thickness = 1
    # text_color = (255, 255, 255)

    # if angles_text:
    #     line_pos = 1
    #     add_text_to_image(
    #         summary_image, f'Knee Angle (Bottom): {knee_angle_bottom:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Knee Angle (Top): {knee_angle_top:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Knee Angle (Avg): {avg_knee_angle:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Shoulder Angle (Avg): {avg_shoulder_angle:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Elbow Angle (Avg): {avg_elbow_angle:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Torso Angle (Avg): {avg_torso_angle:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1
    #     add_text_to_image(
    #         summary_image, f'Ankle Angle (Avg): {avg_ankle_angle:.2f}°', line_pos, text_font_scale, text_thickness)
    #     line_pos += 1

    # --- Draw the angles on the image ---
    # Draw Knee Angle (Hip-Knee-Ankle)
    if frame_landmarks.right_hip and frame_landmarks.right_knee and frame_landmarks.right_ankle:
        knee_angle_val = calculate_angle(
            frame_landmarks.right_hip, frame_landmarks.right_knee, frame_landmarks.right_ankle)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    frame_landmarks.right_hip,
                                                    frame_landmarks.right_knee,
                                                    frame_landmarks.right_ankle,
                                                    knee_angle_val,
                                                    color=(0, 255, 0), arc_radius=80)  # Green

    # Draw Elbow Angle (Shoulder-Elbow-Wrist)
    if frame_landmarks.right_shoulder and frame_landmarks.right_elbow and frame_landmarks.right_wrist:
        elbow_angle_val = calculate_angle(
            frame_landmarks.right_shoulder, frame_landmarks.right_elbow, frame_landmarks.right_wrist)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    frame_landmarks.right_wrist,
                                                    frame_landmarks.right_elbow,
                                                    frame_landmarks.right_shoulder,
                                                    elbow_angle_val,
                                                    color=(255, 0, 255), arc_radius=60)  # Magenta

    # Draw Torso Angle (Shoulder-Hip-Horizontal)
    if frame_landmarks.right_hip and frame_landmarks.right_shoulder:
        horizontal_point = Point(
            frame_landmarks.right_hip.x + 100, frame_landmarks.right_hip.y)
        torso_angle_val = calculate_angle(
            frame_landmarks.right_shoulder, frame_landmarks.right_hip, horizontal_point)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    horizontal_point,
                                                    frame_landmarks.right_hip,
                                                    frame_landmarks.right_shoulder,
                                                    torso_angle_val,
                                                    color=(255, 255, 0), arc_radius=100)  # Yellow

    # Draw Ankle Angle (Knee-Ankle-Foot Index)
    if frame_landmarks.right_knee and frame_landmarks.right_ankle and frame_landmarks.right_foot_index:
        ankle_angle_val = calculate_angle(
            frame_landmarks.right_knee, frame_landmarks.right_ankle, frame_landmarks.right_foot_index)
        summary_image = draw_angle_on_image_precise(summary_image,
                                                    frame_landmarks.right_knee,
                                                    frame_landmarks.right_ankle,
                                                    frame_landmarks.right_foot_index,
                                                    ankle_angle_val,
                                                    color=(0, 255, 255), arc_radius=70)  # Cyan

    plt.figure(figsize=(16, 10))
    plt.imshow(summary_image)
    plt.title(f"Bike Fitting Summary")
    plt.axis('off')
    plt.show()

    image_bgr = cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'cycling_summary.png', image_bgr)
    print(f"Summary image saved as cycling_summary.png")


def show_top_and_bottom_knee_extension(dynamic_angles: dict[str, list[float]], video_path: str):
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
        show_video_at_index(video_path, foot_bottom_index_mp,
                            "Foot at Bottom (Max Knee Extension)")
        show_video_at_index(video_path, foot_top_index_mp,
                            "Foot at Top (Min Knee Extension)")
    else:
        print("\nCould not calculate knee angles with MediaPipe to determine foot bottom/top frames.")


# get the video at a custom index, and apply the landmarks at that index


def get_video_at_index(video_path, index: int, landmarks: AllLandmarks):
    """Retrieves a specific frame from the video and applies landmarks."""
    frame = get_image_at_index(video_path, index)
    if frame is not None:
        # Convert to RGB for plotting
        #        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the landmarks for this index
        if 0 <= index < len(landmarks.frames_landmarks):
            frame_landmarks = landmarks.frames_landmarks[index]
            return frame, frame_landmarks
    return None, None


def show_video_with_landmarks(video_path, index: int, landmarks: AllLandmarks, with_foot_details=False):
    """Displays a specific frame from the video with landmarks."""
    frame_rgb, frame_landmarks = get_video_at_index(
        video_path, index, landmarks)
    if frame_rgb is not None and frame_landmarks is not None:
        plot_image_with_points(frame_rgb, frame_landmarks, with_foot_details)
    else:
        print(f"Could not retrieve or display frame at index {index}")
