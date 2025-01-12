import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import json
import os
from groq import Groq
import csv
import time

class TaskManager:
    def __init__(self):
        self.tasks = [
            {
                "name": "Spread Fingers",
                "instruction": "Spread your fingers as wide as possible",
                "duration": 5,
                "prep_time": 3
            },
            {
                "name": "Hold Still",
                "instruction": "Hold your hand still with fingers relaxed",
                "duration": 5,
                "prep_time": 3
            },
            {
                "name": "Precision Pose",
                "instruction": "Touch your thumb to each fingertip slowly",
                "duration": 10,
                "prep_time": 3
            },
            {
                "name": "Muscle Atrophy Test",
                "instruction": "Squeeze your hand into a fist and hold",
                "duration": 10,
                "prep_time": 3
            }
        ]
        self.current_task = None
        self.task_start_time = None
        self.prep_start_time = None
        self.is_in_prep = False
        self.task_index = 0
        self.is_complete = False

    def save_task_summary(self, file_name="task_summary.txt"):
        """Save the task summary to a text file."""
        try:
            with open(file_name, "w") as file:
                file.write("Task Summary:\n")
                file.write("=" * 50 + "\n")
                for task in self.tasks:
                    file.write(f"Task Name: {task['name']}\n")
                    file.write(f"Instruction: {task['instruction']}\n")
                    file.write(f"Duration: {task['duration']} seconds\n")
                    file.write(f"Preparation Time: {task['prep_time']} seconds\n")
                    file.write("-" * 50 + "\n")
            print(f"Task summary successfully saved to {file_name}")
        except Exception as e:
            print(f"Failed to save task summary: {e}")

    def save_detailed_task_summary(self, metrics_history, health_analyses, file_name="detailed_task_summary.txt"):
        """
        Save a detailed task summary, including only health analyses, to a text file.
        """
        try:
            with open(file_name, "w") as file:
                file.write("DETAILED TASK SUMMARY\n")
                file.write("=" * 50 + "\n\n")
                # Write health analyses
                file.write("Health Analyses:\n")
                for analysis in health_analyses:
                    file.write(analysis)
                    file.write("\n" + "=" * 50 + "\n")
            print(f"Detailed task summary saved to {file_name}")
        except Exception as e:
            print(f"Failed to save detailed task summary: {e}")

    def delete_task_summary(self, file_name="task_summary.txt"):
        """Delete the task summary text file."""
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
                print(f"Task summary file {file_name} deleted successfully.")
            else:
                print(f"Task summary file {file_name} does not exist.")
        except Exception as e:
            print(f"Failed to delete task summary file: {e}")

    def start_next_task(self):
        if self.task_index < len(self.tasks):
            self.current_task = self.tasks[self.task_index]
            self.prep_start_time = time.time()
            self.is_in_prep = True
            self.task_start_time = None
            return True
        self.is_complete = True
        return False

    def get_status(self):
        if self.is_complete:
            return "All tasks completed!", (0, 255, 0)
        if not self.current_task:
            return "Press 'n' to start next task", (255, 255, 255)
        if self.is_in_prep:
            remaining = self.current_task["prep_time"] - (time.time() - self.prep_start_time)
            if remaining <= 0:
                self.is_in_prep = False
                self.task_start_time = time.time()
                return f"START: {self.current_task['instruction']}", (0, 255, 0)
            return f"Get ready: {self.current_task['name']} in {remaining:.1f}s", (0, 255, 255)
        if self.task_start_time:
            elapsed = time.time() - self.task_start_time
            remaining = self.current_task["duration"] - elapsed
            if remaining <= 0:
                self.task_index += 1
                self.current_task = None
                return "Task complete! Press 'n' for next task", (0, 255, 0)
            return f"{remaining:.1f}s remaining", (0, 255, 255)
        return "Press 'n' to start next task", (255, 255, 255)

class HandHealthMonitor:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        # Initialize analysis components
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_period = timedelta(days=7)
        # Initialize metrics storage
        self.metrics_history = []
        # Add position history for better tremor detection
        self.position_history = []
        self.history_length = 30  # Store last 30 frames
        self.csv_file = 'hand_metrics.csv'
        self.csv_fields = ['timestamp', 'finger_spread', 'tremor', 'precision', 'stability', 'tremor_severity']
        self.initialize_csv()
        self.model_trained = False
        # Add muscle atrophy specific metrics
        self.csv_fields.extend(['muscle_volume', 'grip_strength', 'finger_flexion'])
        # Reference measurements for atrophy detection
        self.baseline_measurements = {
            'thumb_length': None,
            'palm_width': None,
            'muscle_volume_baseline': None,
            'grip_strength_baseline': None
        }
        # Atrophy warning thresholds (percentage decrease from baseline)
        self.atrophy_thresholds = {
            'mild': 0.10,  # 10% decrease
            'moderate': 0.20,  # 20% decrease
            'severe': 0.30  # 30% decrease
        }

    def initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_fields + ['task'])
                writer.writeheader()

    def train_model(self):
        """Train the anomaly detection model on collected data"""
        if len(self.metrics_history) < 10:  # Need minimum data points
            return
        df = pd.DataFrame(self.metrics_history)
        features = df[['finger_spread', 'tremor', 'precision', 'stability']]
        scaled_features = self.scaler.fit_transform(features)
        self.anomaly_detector.fit(scaled_features)
        self.model_trained = True

    def save_metrics_to_csv(self, metrics, task_name):
        """Save the collected metrics to a CSV file and retrain model"""
        metrics['task'] = task_name  # Add task name to metrics
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.csv_fields + ['task'])
            writer.writerow(metrics)
        # Retrain model with new data
        self.train_model()

    def calculate_tremor_intensity(self, current_positions):
        """Calculate tremor intensity using position history"""
        self.position_history.append(current_positions)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        if len(self.position_history) < 2:
            return 0.0
        # Calculate position changes between consecutive frames
        changes = []
        for i in range(1, len(self.position_history)):
            prev = np.array(self.position_history[i-1])
            curr = np.array(self.position_history[i])
            change = np.abs(curr - prev).mean()
            changes.append(change)
        # Higher weights for more recent changes
        weights = np.linspace(0.5, 1.0, len(changes))
        weighted_changes = np.array(changes) * weights
        # Calculate tremor intensity
        tremor = np.mean(weighted_changes) * 100  # Scale up for better visibility
        return tremor

    def calculate_metrics(self, landmarks, task_name):
        """Calculate key metrics from hand landmarks with improved tremor detection"""
        # Get current positions
        current_positions = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
        # Calculate enhanced tremor
        tremor = self.calculate_tremor_intensity(current_positions)
        # Calculate finger spread (distance between thumb and pinky)
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        spread = np.sqrt((thumb_tip.x - pinky_tip.x)**2 +
                         (thumb_tip.y - pinky_tip.y)**2)
        # Calculate movement precision with more weight on z-axis stability
        z_positions = [lm.z for lm in landmarks.landmark]
        precision = 1.0 - (np.std(z_positions) * 2)  # Double weight on z-axis variance
        # Calculate fingertip stability
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        fingertip_positions = np.array([[landmarks.landmark[tip].x,
                                         landmarks.landmark[tip].y,
                                         landmarks.landmark[tip].z] for tip in fingertips])
        stability = 1.0 - np.var(fingertip_positions).mean()
        # Calculate muscle metrics
        muscle_metrics = self.calculate_muscle_metrics(landmarks)
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'finger_spread': float(spread),  # in arbitrary units
            'tremor': float(tremor),  # in arbitrary units
            'precision': float(precision),  # unitless
            'stability': float(stability),  # unitless
            'tremor_severity': self.classify_tremor(tremor),
            'muscle_volume': muscle_metrics['muscle_volume'],  # in arbitrary units
            'grip_strength': muscle_metrics['grip_strength'],  # unitless
            'finger_flexion': muscle_metrics['finger_flexion']  # in radians
        }
        self.metrics_history.append(metrics)
        self.save_metrics_to_csv(metrics, task_name)
        return metrics

    def classify_tremor(self, tremor_value):
        """Classify tremor severity with adjusted thresholds"""
        if tremor_value < 0.2:
            return "Minimal"
        elif tremor_value < 0.5:
            return "Mild"
        elif tremor_value < 0.8:
            return "Moderate"
        else:
            return "Severe"

    def process_frame(self, frame, task_name):
        """Process a single frame and return metrics"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        metrics = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # Calculate metrics
                metrics = self.calculate_metrics(hand_landmarks, task_name)
                # Add metrics display to frame
                if metrics:
                    self.display_metrics(frame, metrics)
        return frame, metrics

    def display_metrics(self, frame, metrics, task_status=None, task_color=(255, 255, 255)):
        """Display metrics on the frame with color-coded tremor severity"""
        height, width = frame.shape[:2]
        # Define color based on tremor severity
        tremor_colors = {
            "Minimal": (0, 255, 0),    # Green
            "Mild": (0, 255, 255),     # Yellow
            "Moderate": (0, 165, 255), # Orange
            "Severe": (0, 0, 255)      # Red
        }
        color = tremor_colors.get(metrics['tremor_severity'], (0, 255, 0))
        # Display task status at the top
        if task_status:
            cv2.putText(frame, task_status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, task_color, 2)
        # Display task instruction if available
        if self.task_manager.current_task:
            instruction = self.task_manager.current_task['instruction']
            cv2.putText(frame, instruction, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Display metrics below task status
        metrics_text = [
            f"Tremor: {metrics['tremor']:.3f} ({metrics['tremor_severity']})",
            f"Precision: {metrics['precision'] * 100:.2f}% ",
            f"Stability: {metrics['stability'] * 100:.2f}% ",
            f"Finger Spread: {metrics['finger_spread'] * 1000:.2f} mm",
            f"Muscle Volume: {metrics['muscle_volume'] * 1000:.2f} mm^3",
            f"Grip Strength: {metrics['grip_strength'] * 10:.2f} Newtons",
            f"Finger Flexion: {np.degrees(metrics['finger_flexion']):.2f} degrees"
        ]
        for i, text in enumerate(metrics_text):
            cv2.putText(frame, text, (10, 100 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def analyze_health_trends(self):
        """Analyze collected metrics for health trends"""
        if not self.model_trained:
            return "Model not trained yet"
        if len(self.metrics_history) < 10:  # Need minimum data points
            return None
        df = pd.DataFrame(self.metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Calculate basic statistics from recent data
        recent_data = df.tail(min(100, len(df)))
        stats = {
            'tremor_level': recent_data['tremor'].mean(),
            'tremor_max': recent_data['tremor'].max(),
            'tremor_min': recent_data['tremor'].min(),
            'precision_score': recent_data['precision'].mean(),
            'stability_score': recent_data['stability'].mean(),
            'spread_consistency': recent_data['finger_spread'].std(),
            'tremor_severity': recent_data['tremor_severity'].mode().iloc[0] if 'tremor_severity' in recent_data else 'Unknown',
            'data_points': len(df),
            'duration_minutes': (pd.to_datetime(recent_data['timestamp'].iloc[-1]) -
                                 pd.to_datetime(recent_data['timestamp'].iloc[0])).total_seconds() / 60
        }
        # Add trend analysis
        if len(recent_data) >= 2:
            stats['tremor_trend'] = (recent_data['tremor'].iloc[-1] -
                                     recent_data['tremor'].iloc[0])
            stats['precision_trend'] = (recent_data['precision'].iloc[-1] -
                                        recent_data['precision'].iloc[0])
        return stats

    def calculate_muscle_metrics(self, landmarks):
        """Calculate metrics related to muscle volume and strength"""
        # Calculate approximate muscle volume using hand landmarks
        palm_width = self.calculate_palm_width(landmarks)
        thenar_volume = self.estimate_thenar_volume(landmarks)
        hypothenar_volume = self.estimate_hypothenar_volume(landmarks)
        # Estimate grip strength based on finger positioning
        grip_strength = self.estimate_grip_strength(landmarks)
        # Calculate finger flexion angles
        finger_flexion = self.calculate_finger_flexion(landmarks)
        return {
            'muscle_volume': thenar_volume + hypothenar_volume,
            'grip_strength': grip_strength,
            'finger_flexion': finger_flexion
        }

    def calculate_palm_width(self, landmarks):
        """Calculate palm width using landmarks"""
        thumb_cmc = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        return np.sqrt((thumb_cmc.x - pinky_mcp.x)**2 +
                       (thumb_cmc.y - pinky_mcp.y)**2)

    def estimate_thenar_volume(self, landmarks):
        """Estimate thenar eminence volume"""
        thumb_cmc = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]
        thumb_mcp = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        # Calculate approximate volume using landmark positions
        length = np.sqrt((thumb_cmc.x - thumb_mcp.x)**2 +
                         (thumb_cmc.y - thumb_mcp.y)**2)
        width = np.sqrt((thumb_cmc.x - wrist.x)**2 +
                        (thumb_cmc.y - wrist.y)**2)
        return length * width * width  # Approximate volume

    def estimate_hypothenar_volume(self, landmarks):
        """Estimate hypothenar muscle volume"""
        pinky_mcp = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        # Calculate approximate volume
        length = np.sqrt((pinky_mcp.x - wrist.x)**2 +
                         (pinky_mcp.y - wrist.y)**2)
        return length * length * length * 0.5  # Approximate volume

    def estimate_grip_strength(self, landmarks):
        """Estimate relative grip strength based on finger positions"""
        # Calculate distances between fingertips and palm
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        palm_center = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        distances = []
        for tip in fingertips:
            tip_landmark = landmarks.landmark[tip]
            distance = np.sqrt((tip_landmark.x - palm_center.x)**2 +
                               (tip_landmark.y - palm_center.y)**2)
            distances.append(distance)
        # Normalize and convert to strength estimate
        return 1.0 - (np.mean(distances) / self.calculate_palm_width(landmarks))

    def calculate_finger_flexion(self, landmarks):
        """Calculate average finger flexion angles"""
        fingers = [
            [self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
             self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
             self.mp_hands.HandLandmark.INDEX_FINGER_DIP],
            [self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
             self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
             self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP],
            [self.mp_hands.HandLandmark.RING_FINGER_MCP,
             self.mp_hands.HandLandmark.RING_FINGER_PIP,
             self.mp_hands.HandLandmark.RING_FINGER_DIP],
            [self.mp_hands.HandLandmark.PINKY_MCP,
             self.mp_hands.HandLandmark.PINKY_PIP,
             self.mp_hands.HandLandmark.PINKY_DIP]
        ]
        flexion_angles = []
        for finger in fingers:
            mcp = landmarks.landmark[finger[0]]
            pip = landmarks.landmark[finger[1]]
            dip = landmarks.landmark[finger[2]]
            # Calculate angles between segments
            angle1 = np.arctan2(pip.y - mcp.y, pip.x - mcp.x)
            angle2 = np.arctan2(dip.y - pip.y, dip.x - pip.x)
            flexion_angle = np.abs(angle2 - angle1)
            flexion_angles.append(flexion_angle)
        return np.mean(flexion_angles)

class GroqAnalyzer:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def analyze_health_trends(self, metrics_summary):
        """Use Groq to analyze health trends with enhanced tremor focus"""
        if not metrics_summary:
            return "Insufficient data for analysis"
        prompt = f"""
        You are a specialized medical AI focusing on astronaut hand movement analysis.
        Analyze these hand movement metrics with particular attention to tremors:
        Metrics Summary:
        {metrics_summary}
        Rules for analysis:
        1. Tremor values above 0.5 should ALWAYS be flagged as concerning
        2. Any 'Moderate' or 'Severe' tremor classification requires immediate attention
        3. Consider stability metrics in conjunction with tremor values
        4. Look for patterns in precision degradation
        Please provide:
        1. IMMEDIATE HEALTH STATUS:
           - Current tremor severity and its implications
           - Whether immediate action is needed
        2. DETAILED ANALYSIS:
           - Specific tremor characteristics observed
           - Correlation with other metrics
           - Changes from baseline if available
        3. RECOMMENDATIONS:
           - Specific exercises or interventions based on tremor type
           - Whether medical consultation is needed
           - Monitoring frequency adjustments
        4. RISK ASSESSMENT:
           - Current risk level (Low/Medium/High)
           - Potential complications if untreated
           - Timeline for recommended interventions
        """
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error getting Groq analysis: {str(e)}"

def main():
    # Initialize monitoring system
    monitor = HandHealthMonitor()
    task_manager = TaskManager()
    monitor.task_manager = task_manager  # Link task manager to monitor

    # Save the task summary at the start
    try:
        task_manager.save_task_summary()
    except AttributeError as e:
        print(f"Error: {e}. Ensure the `save_task_summary` method exists in the TaskManager class.")
        return

    # Initialize Groq analyzer if API key is available
    groq_api_key = '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'
    groq_analyzer = GroqAnalyzer(groq_api_key) if groq_api_key else None

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera. Please check your device.")
        return

    frame_count = 0
    analysis_interval = 50  # Analyze more frequently (every 50 frames)
    retrain_interval = 500  # Retrain model every 500 frames
    health_analyses = []  # Store all health analyses

    print("Starting enhanced hand movement monitoring...")
    print("Press 'n' to start tasks")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        task_name = task_manager.current_task['name'] if task_manager.current_task else "No Task"
        frame, metrics = monitor.process_frame(frame, task_name)
        frame_count += 1

        # Get task status
        status, color = task_manager.get_status()
        if metrics:
            monitor.display_metrics(frame, metrics, status, color)

        cv2.imshow('Astronaut Hand Health Monitoring', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Quitting the monitoring system...")
            break
        elif key == ord('n'):
            if not task_manager.current_task:
                task_manager.start_next_task()

        # Only record metrics during active tasks (not during prep time)
        if task_manager.current_task and not task_manager.is_in_prep and metrics:
            monitor.metrics_history.append(metrics)

        # Periodic model retraining
        if frame_count % retrain_interval == 0:
            try:
                monitor.train_model()
            except Exception as e:
                print(f"Error during model retraining: {e}")

        # Periodic health trend analysis
        if frame_count % analysis_interval == 0 and groq_analyzer:
            health_trends = monitor.analyze_health_trends()
            if health_trends:
                print("\nAnalyzing health trends...")
                analysis = groq_analyzer.analyze_health_trends(health_trends)
                health_analyses.append(analysis)
                print("\nHealth Analysis:")
                print(analysis)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Save final metrics and analyses
    if monitor.metrics_history:
        try:
            with open('hand_metrics.json', 'w') as f:
                json.dump(monitor.metrics_history, f, indent=2)
            print("\nMetrics saved to hand_metrics.json")
            print(f"\nMetrics also saved to {monitor.csv_file}")
        except Exception as e:
            print(f"Error saving metrics: {e}")

        if groq_analyzer:
            try:
                task_manager.save_detailed_task_summary(
                    monitor.metrics_history, health_analyses, file_name="detailed_task_summary.txt"
                )
            except Exception as e:
                print(f"Error saving detailed task summary: {e}")

    # Delete the task summary file at the end
    task_manager.delete_task_summary()

    print("Monitoring session ended.")

if __name__ == "__main__":
    main()



