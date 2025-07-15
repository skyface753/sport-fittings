
from src.angles_functions import calculate_seat_height_adjustment
import numpy as np


def get_optimal_ranges(angle_label: str, mode: str, angle_specs) -> list[tuple[int, int]]:
    # Find spec matching label and mode
    spec = next((s for s in angle_specs if s.label ==
                angle_label and mode in s.modes), None)
    if spec:
        return spec.optimal_ranges.get(mode, [])
    return []


def get_bike_fit_recommendations(angles_data: dict[str, list[float]],
                                 avg_knee_angle_peaks_high: float,
                                 avg_knee_angle_peaks_low: float,
                                 kops_value: float,
                                 mode: str,
                                 angle_specs,
                                 seat_height_sensitive_factor: int = 4) -> list[str]:
    """
    Generates bike fit recommendations based on calculated angles and standard ranges.
    """
    recommendations = []
    recommendations.append("--- Bike Fit Recommendations ---")

    # Helper for avg angle or 0 if no data
    def avg_angle(label):
        return np.mean(angles_data.get(label, [0]))

    # Knee bottom extension ranges: assume first range in list
    knee_bottom_ranges = get_optimal_ranges(
        "Knee Angle (Hip-Knee-Ankle)", mode, angle_specs)
    # We expect the first range to be bottom, second top — if both present
    knee_bottom_range = knee_bottom_ranges[0] if len(
        knee_bottom_ranges) > 0 else None
    knee_top_range = knee_bottom_ranges[1] if len(
        knee_bottom_ranges) > 1 else None

    # Torso angle
    torso_ranges = get_optimal_ranges(
        "Torso Angle (Shoulder-Hip-Horizontal)", mode, angle_specs)
    torso_range = torso_ranges[0] if torso_ranges else None

    # Elbow angle
    elbow_ranges = get_optimal_ranges(
        "Elbow Angle (Shoulder-Elbow-Wrist)", mode, angle_specs)
    elbow_range = elbow_ranges[0] if elbow_ranges else None

    # Hip angle (only for aeros)
    hip_ranges = get_optimal_ranges(
        "Hip Angle (Shoulder-Hip-Knee)", mode, angle_specs)
    hip_range = hip_ranges[0] if hip_ranges else None

    # 1. Saddle Height (Knee Bottom)
    recommendations.append(
        f"Your Knee Angle (Hip-Knee-Ankle) at Bottom of Stroke: {avg_knee_angle_peaks_high:.2f}")
    if knee_bottom_range:
        if avg_knee_angle_peaks_high < knee_bottom_range[0]:
            recommendations.append(
                f"• Saddle might be TOO LOW. Raise saddle height. (Optimal: {knee_bottom_range[0]}-{knee_bottom_range[1]}°)")
            recommendations.append(
                f"• Maybe try: {calculate_seat_height_adjustment(current_angle=avg_knee_angle_peaks_high, next_optimal_angle=knee_bottom_range[0], sensitivity_factor=seat_height_sensitive_factor)} cm higher saddle.")
        elif avg_knee_angle_peaks_high > knee_bottom_range[1]:
            recommendations.append(
                f"• Saddle might be TOO HIGH. Lower saddle height. (Optimal: {knee_bottom_range[0]}-{knee_bottom_range[1]}°)")
            # calculate_seat_height_adjustment(current_angle=knee_bottom_range[1], optimal_range=knee_bottom_range, sensitivity_factor=seat_height_sensitive_factor)
            recommendations.append(
                f"• Maybe try: {calculate_seat_height_adjustment(current_angle=avg_knee_angle_peaks_high, next_optimal_angle=knee_bottom_range[1], sensitivity_factor=seat_height_sensitive_factor)} cm lower saddle.")
        else:
            recommendations.append(
                f"• Knee extension (bottom): Within optimal range. ({knee_bottom_range[0]}-{knee_bottom_range[1]}°)")

    # 2. Knee Top extension
    recommendations.append(
        f"Your Knee Angle (Hip-Knee-Ankle) at Top of Stroke: {avg_knee_angle_peaks_low:.2f}")
    if knee_top_range:
        if avg_knee_angle_peaks_low < knee_top_range[0]:
            recommendations.append(
                f"• Knee extension (top) too bent. Raise saddle or adjust cleat. (Optimal: {knee_top_range[0]}-{knee_top_range[1]}°)")
        elif avg_knee_angle_peaks_low > knee_top_range[1]:
            recommendations.append(
                f"• Knee extension (top) too straight. Lower saddle or adjust cleat. (Optimal: {knee_top_range[0]}-{knee_top_range[1]}°)")
        else:
            recommendations.append(
                f"• Knee extension (top): Within optimal range. ({knee_top_range[0]}-{knee_top_range[1]}°)")

    # 3. Torso Angle
    avg_torso_angle = avg_angle("Torso Angle (Shoulder-Hip-Horizontal)")
    recommendations.append(
        f"Your Torso Angle (Shoulder-Hip-Horizontal) AVG: {avg_torso_angle:.2f}")
    if torso_range:
        if avg_torso_angle < torso_range[0]:
            recommendations.append(
                f"• Torso angle too aggressive. Raise handlebars or shorter stem. (Optimal: {torso_range[0]}-{torso_range[1]}°)")
        elif avg_torso_angle > torso_range[1]:
            recommendations.append(
                f"• Torso angle too upright. Lower handlebars or longer stem. (Optimal: {torso_range[0]}-{torso_range[1]}°)")
        else:
            recommendations.append(
                f"• Torso angle: Within optimal range. ({torso_range[0]}-{torso_range[1]}°)")

    # 4. Elbow Angle
    avg_elbow_angle = avg_angle("Elbow Angle (Shoulder-Elbow-Wrist)")
    recommendations.append(
        f"Your Elbow Angle (Shoulder-Elbow-Wrist) AVG: {avg_elbow_angle:.2f}")
    if elbow_range:
        if avg_elbow_angle > elbow_range[1]:
            recommendations.append(
                f"• Elbows too straight. Slight bend recommended. (Optimal: {elbow_range[0]}-{elbow_range[1]}°)")
        elif avg_elbow_angle < elbow_range[0]:
            recommendations.append(
                f"• Elbows too bent. Consider lengthening reach. (Optimal: {elbow_range[0]}-{elbow_range[1]}°)")
        else:
            recommendations.append(
                f"• Elbow angle: Within optimal range. ({elbow_range[0]}-{elbow_range[1]}°)")

    # 5. Hip Angle (if aeros mode)
    if hip_range:
        avg_hip_angle = avg_angle("Hip Angle (Shoulder-Hip-Knee)")
        recommendations.append(
            f"Your Hip Angle (Shoulder-Hip-Knee) AVG: {avg_hip_angle:.2f}")
        if avg_hip_angle < hip_range[0]:
            recommendations.append(
                f"• Hip angle too closed. Raise saddle or adjust cleat. (Optimal: {hip_range[0]}-{hip_range[1]}°)")
        elif avg_hip_angle > hip_range[1]:
            recommendations.append(
                f"• Hip angle too open. Lower saddle or adjust cleat. (Optimal: {hip_range[0]}-{hip_range[1]}°)")
        else:
            recommendations.append(
                f"• Hip angle: Within optimal range. ({hip_range[0]}-{hip_range[1]}°)")
    else:
        recommendations.append(
            "Hip angle analysis not applicable for current mode.")

    # 6. KOPS Analysis
    if kops_value is not None:
        recommendations.append(
            f"KOPS horizontal distance (at max foot X): {kops_value:.2f} pixels")
        if kops_value < 0:
            recommendations.append(
                "• Negative value: Knee is in front of estimated pedal spindle (foot index).")
        elif kops_value > 0:
            recommendations.append(
                "• Positive value: Knee is behind estimated pedal spindle (foot index).")
        else:
            recommendations.append(
                "• KOPS value is zero: Knee is aligned with estimated pedal spindle (foot index).")
    else:
        recommendations.append("KOPS analysis not performed or failed.")

    # General advice
    recommendations.append("\n--- General Advice ---")
    recommendations.append(
        "• These are general recommendations. Individual comfort and flexibility are key.")
    recommendations.append(
        "• Consult a professional bike fitter for personalized adjustments.")
    recommendations.append(
        "• Make small adjustments and test them on the bike.")

    return recommendations

    # Get average angles
    # avg_knee_angle = np.mean(angles_data.get(
    #     "Knee Angle (Hip-Knee-Ankle)", [0]))
    # avg_torso_angle = np.mean(angles_data.get(
    #     "Torso Angle (Shoulder-Hip-Horizontal)", [0]))
    # avg_elbow_angle = np.mean(angles_data.get(
    #     "Elbow Angle (Shoulder-Elbow-Wrist)", [0]))
    # avg_knee_angle = avg_knee_angle if avg_knee_angle <= 90 else 180 - avg_knee_angle
    # avg_torso_angle = avg_torso_angle if avg_torso_angle <= 90 else 180 - avg_torso_angle

    optimal_knee_extension_peaks = optimal_angles.knee_extension_bottom
    optimal_knee_extension_peaks_low = optimal_angles.knee_extension_top
    optimal_torso_angle_to_horizontal = optimal_angles.torso_to_horizontal
    optimal_elbow_angle = optimal_angles.elbow_angle
    # check if the angles are from drops or aeros
    if isinstance(optimal_angles, OptimalAnglesOnTheAeros):
        optimal_hip_angle = optimal_angles.hip_angle

    # 1. Saddle Height (based on Knee Extension at bottom of stroke)
    recommendations.append(
        f"Your Knee Angle (Hip-Knee-Ankle) at Bottom of Stroke: {avg_knee_angle_peaks_high:.2f}")
    if avg_knee_angle_peaks_high < optimal_knee_extension_peaks[0]:
        recommendations.append(
            f"• Saddle might be TOO LOW. Consider RAISING your saddle height. (Optimal: {optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")
    elif avg_knee_angle_peaks_high > optimal_knee_extension_peaks[1]:
        recommendations.append(
            f"• Saddle might be TOO HIGH. Consider LOWERING your saddle height. (Optimal: {optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")
    else:
        recommendations.append(
            f"• Knee extension (bottom): Within optimal range. ({optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")

    # 2. Knee Extension at Top of Stroke
    recommendations.append(
        f"Your Knee Angle (Hip-Knee-Ankle) at Top of Stroke: {avg_knee_angle_peaks_low:.2f}")
    if avg_knee_angle_peaks_low < optimal_knee_extension_peaks_low[0]:
        recommendations.append(
            f"• Knee extension (top) is TOO BENT. Consider RAISING saddle height or adjusting cleat position. (Optimal: {optimal_knee_extension_peaks_low[0]}-{optimal_knee_extension_peaks_low[1]}°)")
    elif avg_knee_angle_peaks_low > optimal_knee_extension_peaks_low[1]:
        recommendations.append(
            f"• Knee extension (top) is TOO STRAIGHT. Consider LOWERING saddle height or adjusting cleat position. (Optimal: {optimal_knee_extension_peaks_low[0]}-{optimal_knee_extension_peaks_low[1]}°)")
    else:
        recommendations.append(
            f"• Knee extension (top): Within optimal range. ({optimal_knee_extension_peaks_low[0]}-{optimal_knee_extension_peaks_low[1]}°)")

    # 2. Saddle Fore/Aft (less direct from 2D angle, but can be inferred)
    # IS CALCULATED SEPERATE
    # recommendations.append("• **Saddle fore/aft position** requires further analysis (e.g., KOPS - Knee Over Pedal Spindle).")

    # 3. Handlebar Reach / Torso Angle
    recommendations.append(
        f"Your Torso Angle (Shoulder-Hip-Horizontal) AVG: {avg_torso_angle:.2f}")
    if avg_torso_angle < optimal_torso_angle_to_horizontal[0]:
        recommendations.append(
            f"• Torso angle is TOO AGGRESSIVE. Consider RAISING handlebars or getting a shorter stem. (Optimal: {optimal_torso_angle_to_horizontal[0]}-{optimal_torso_angle_to_horizontal[1]}°)")
    elif avg_torso_angle > optimal_torso_angle_to_horizontal[1]:
        recommendations.append(
            f"• Torso angle is TOO UPRIGHT. Consider LOWERING handlebars or getting a longer stem (if more aggressive stance desired). (Optimal: {optimal_torso_angle_to_horizontal[0]}-{optimal_torso_angle_to_horizontal[1]}°)")
    else:
        recommendations.append(
            f"• Torso angle: Within optimal range. ({optimal_torso_angle_to_horizontal[0]}-{optimal_torso_angle_to_horizontal[1]}°)")

    # 4. Handlebar Width / Elbow Angle / Arm Bend
    recommendations.append(
        f"Your Elbow Angle (Shoulder-Elbow-Wrist) AVG: {avg_elbow_angle:.2f}")
    if avg_elbow_angle > optimal_elbow_angle[1]:
        recommendations.append(
            f"• Elbows appear TOO STRAIGHT. Try to maintain a slight bend for comfort and control. This could indicate a reach issue or stiff arms. (Optimal: {optimal_elbow_angle[0]}-{optimal_elbow_angle[1]}°)")
    elif avg_elbow_angle < optimal_elbow_angle[0]:
        recommendations.append(
            f"• Elbows are TOO BENT. This might indicate reach is too short or an overly aggressive position, consider lengthening reach. (Optimal: {optimal_elbow_angle[0]}-{optimal_elbow_angle[1]}°)")
    else:
        recommendations.append(
            f"• Elbow angle: Within optimal range. ({optimal_elbow_angle[0]}-{optimal_elbow_angle[1]}°)")

    # 5. Hip Angle (if on aeros)
    if isinstance(optimal_angles, OptimalAnglesOnTheAeros):
        hip_angle = angles_data.get("Hip Angle (Shoulder-Hip-Knee)", [])
        if hip_angle:
            avg_hip_angle = np.mean(hip_angle)
            recommendations.append(
                f"Your Hip Angle (Shoulder-Hip-Elbow) AVG: {avg_hip_angle:.2f}")
            if avg_hip_angle < optimal_hip_angle[0]:
                recommendations.append(
                    f"• Hip angle is TOO CLOSED. Consider RAISING saddle or adjusting cleat position. (Optimal: {optimal_hip_angle[0]}-{optimal_hip_angle[1]}°)")
            elif avg_hip_angle > optimal_hip_angle[1]:
                recommendations.append(
                    f"• Hip angle is TOO OPEN. Consider LOWERING saddle or adjusting cleat position. (Optimal: {optimal_hip_angle[0]}-{optimal_hip_angle[1]}°)")
            else:
                recommendations.append(
                    f"• Hip angle: Within optimal range. ({optimal_hip_angle[0]}-{optimal_hip_angle[1]}°)")
        else:
            recommendations.append(
                "• Hip angle data not available for aeros position.")
    else:
        recommendations.append(
            "Hip angle analysis is not applicable, when not in aeros position.")

    # Add general advice
    recommendations.append("\n--- General Advice ---")
    recommendations.append(
        "• These are general recommendations. Individual comfort and flexibility are key.")
    recommendations.append(
        "• Consider consulting a professional bike fitter for personalized adjustments.")
    recommendations.append(
        "• Make small adjustments and test them on the bike.")

    return recommendations
