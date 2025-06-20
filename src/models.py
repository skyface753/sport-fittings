import mediapipe as mp


class Point:
    """Represents a 2D point with x and y coordinates."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        # Round to 2 decimal places
        return f"({self.x:.2f}, {self.y:.2f})"

    def __sub__(self, other):
        """Subtracts another Point from this Point, returning a new Point (vector)."""
        return Point(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        """Adds two Points, returning a new Point (vector)."""
        return Point(self.x + other.x, self.y + other.y)

    def get_from_gpu(self):
        """Attempts to move tensor coordinates from GPU to CPU if they are on GPU."""
        if hasattr(self.x, 'cpu'):
            self.x = self.x.cpu().item()  # .item() to get scalar value
        if hasattr(self.y, 'cpu'):
            self.y = self.y.cpu().item()  # .item() to get scalar value


class FrameLandmarks:
    """Stores landmarks for a single frame."""

    def __init__(self):
        self.nose: Point = None
        self.right_wrist: Point = None
        self.right_elbow: Point = None
        self.right_shoulder: Point = None
        self.right_hip: Point = None
        self.right_knee: Point = None
        self.right_ankle: Point = None
        self.right_foot_index: Point = None
        self.right_heel: Point = None
        # Add left side for potential future use (e.g., bilateral analysis)
        self.left_wrist: Point = None
        self.left_elbow: Point = None
        self.left_shoulder: Point = None
        self.left_hip: Point = None
        self.left_knee: Point = None
        self.left_ankle: Point = None
        self.left_foot_index: Point = None
        self.left_heel: Point = None

    def from_gpu_to_cpu(self):
        """Converts all Point objects within this frame's landmarks from GPU to CPU."""
        for attr_name in dir(self):
            if isinstance(getattr(self, attr_name), Point):
                getattr(self, attr_name).get_from_gpu()
                
    def __str__(self): 
        """String representation of the frame landmarks."""
        return (f"Nose: {self.nose}, Right Wrist: {self.right_wrist}, "
                f"Right Elbow: {self.right_elbow}, Right Shoulder: {self.right_shoulder}, "
                f"Right Hip: {self.right_hip}, Right Knee: {self.right_knee}, "
                f"Right Ankle: {self.right_ankle}, Right Foot Index: {self.right_foot_index}, "
                f"Right Heel: {self.right_heel}, Left Wrist: {self.left_wrist}, "
                f"Left Elbow: {self.left_elbow}, Left Shoulder: {self.left_shoulder}, "
                f"Left Hip: {self.left_hip}, Left Knee: {self.left_knee}, "
                f"Left Ankle: {self.left_ankle}, Left Foot Index: {self.left_foot_index}, "
                f"Left Heel: {self.left_heel}")


class AllLandmarks:
    """Stores a list of FrameLandmarks for all processed frames."""

    def __init__(self):
        self.frames_landmarks: list[FrameLandmarks] = []

    def append_frame(self, frame_landmarks: FrameLandmarks):
        self.frames_landmarks.append(frame_landmarks)

    def get_landmark_series(self, landmark_name: str) -> list[Point]:
        """Returns a list of Points for a specific landmark across all frames."""
        series = []
        for frame_lm in self.frames_landmarks:
            if hasattr(frame_lm, landmark_name) and getattr(frame_lm, landmark_name) is not None:
                series.append(getattr(frame_lm, landmark_name))
        return series

    def clear(self):
        """Clears all stored landmarks."""
        self.frames_landmarks = []


class MPLandmark:
    NOSE = mp.solutions.pose.PoseLandmark.NOSE
    RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
    RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
    RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    RIGHT_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP
    RIGHT_KNEE = mp.solutions.pose.PoseLandmark.RIGHT_KNEE
    RIGHT_ANKLE = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
    RIGHT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX
    RIGHT_HEEL = mp.solutions.pose.PoseLandmark.RIGHT_HEEL
    # Add left side for completeness
    LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST
    LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
    LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP
    LEFT_KNEE = mp.solutions.pose.PoseLandmark.LEFT_KNEE
    LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE
    LEFT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX
    LEFT_HEEL = mp.solutions.pose.PoseLandmark.LEFT_HEEL
