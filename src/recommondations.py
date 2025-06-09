
from src.optimal_angles import OptimalAngles
import numpy as np


def get_bike_fit_recommendations(angles_data: dict[str, list[float]],
                                 knee_angle_peaks_avg: float) -> list[str]:
    """
    Generates bike fit recommendations based on calculated angles and standard ranges.
    """
    recommendations = []

    optimal_angles = OptimalAngles()

    # Get average angles
    avg_knee_angle = np.mean(angles_data.get(
        "Knee Angle (Hip-Knee-Ankle)", [0]))
    avg_torso_angle = np.mean(angles_data.get(
        "Torso Angle (Shoulder-Hip-Horizontal)", [0]))
    avg_elbow_angle = np.mean(angles_data.get(
        "Elbow Angle (Shoulder-Elbow-Wrist)", [0]))
    # avg_knee_angle = avg_knee_angle if avg_knee_angle <= 90 else 180 - avg_knee_angle
    # avg_torso_angle = avg_torso_angle if avg_torso_angle <= 90 else 180 - avg_torso_angle

    recommendations.append("--- Bike Fit Recommendations ---")

    optimal_knee_extension_peaks = optimal_angles.knee_extension_bottom
    optimal_torso_angle_to_horizontal = optimal_angles.torso_to_horizontal
    optimal_elbow_angle = optimal_angles.elbow_angle

    # 1. Saddle Height (based on Knee Extension at bottom of stroke)
    recommendations.append(
        f"Your Knee Angle (Hip-Knee-Ankle) in the Peaks: {knee_angle_peaks_avg:.2f}")
    if knee_angle_peaks_avg < optimal_knee_extension_peaks[0]:
        recommendations.append(
            f"• Saddle might be TOO LOW. Consider RAISING your saddle height. (Optimal: {optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")
    elif knee_angle_peaks_avg > optimal_knee_extension_peaks[1]:
        recommendations.append(
            f"• Saddle might be TOO HIGH. Consider LOWERING your saddle height. (Optimal: {optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")
    else:
        recommendations.append(
            f"• Knee extension (bottom): Within optimal range. ({optimal_knee_extension_peaks[0]}-{optimal_knee_extension_peaks[1]}°)")

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

    # Add general advice
    recommendations.append("\n--- General Advice ---")
    recommendations.append(
        "• These are general recommendations. Individual comfort and flexibility are key.")
    recommendations.append(
        "• Consider consulting a professional bike fitter for personalized adjustments.")
    recommendations.append(
        "• Make small adjustments and test them on the bike.")

    return recommendations
