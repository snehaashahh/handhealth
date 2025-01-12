Hand Health Monitoring and Task Management System
This project provides a comprehensive system for hand health monitoring and rehabilitation, combining computer vision, machine learning, and task management features to assess and track hand performance metrics. The system is designed to support rehabilitation exercises, detect anomalies in hand movements, and analyze muscle atrophy over time. It integrates advanced hand-tracking using MediaPipe and provides real-time feedback to users.
Key Features
1. Hand Tracking and Analysis
Utilizes MediaPipe for real-time hand landmark detection.
Tracks key metrics such as finger spread, tremor intensity, precision, stability, and muscle volume.
Supports enhanced tremor detection with weighted historical frame data.
2. Task Management
Built-in task manager with predefined rehabilitation tasks:
Spread Fingers
Hold Still
Precision Pose
Muscle Atrophy Test
Provides task instructions, preparation time, and duration tracking.
Displays real-time task progress and feedback.
3. Machine Learning for Anomaly Detection
Employs an Isolation Forest model to detect anomalies in hand metrics.
Scales and trains the model on user-specific historical data for personalized feedback.
4. Muscle Atrophy Detection
Tracks changes in muscle volume, grip strength, and finger flexion over time.
Detects and categorizes atrophy severity (mild, moderate, or severe) based on user-specific baseline measurements.
5. Data Storage and Visualization
Metrics are logged to a CSV file for further analysis and monitoring.
Includes timestamps and task-specific data for tracking progress.
6. User-Friendly Interface
Visualizes hand landmarks and metrics in real-time using OpenCV.
Displays task status, metrics, and tremor severity with color-coded feedback.
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/username/hand-health-monitor.git
cd hand-health-monitor


Install dependencies:
bash
Copy code
pip install -r requirements.txt


Usage
Run the program:
bash
Copy code
python hand_health_monitor.py


Use the interface to perform rehabilitation tasks:
Press n to start the next task.
Follow on-screen instructions and feedback during tasks.
Dependencies
Python 3.8+
MediaPipe
OpenCV
NumPy
Pandas
scikit-learn
datetime
CSV
Groq (optional)
Metrics Calculated
Finger Spread: Measures the distance between thumb and pinky fingertips.
Tremor Intensity: Analyzes hand stability across frames.
Precision: Assesses movement smoothness, emphasizing z-axis stability.
Stability: Evaluates fingertip stability.
Muscle Volume: Tracks potential signs of muscle atrophy.
Grip Strength: Estimates hand strength based on task performance.
Finger Flexion: Tracks finger bending and positioning.
Tasks
1. Spread Fingers
Instruction: Spread fingers as wide as possible.
Duration: 5 seconds
Preparation Time: 3 seconds
2. Hold Still
Instruction: Keep the hand still with relaxed fingers.
Duration: 5 seconds
Preparation Time: 3 seconds
3. Precision Pose
Instruction: Touch the thumb to each fingertip slowly.
Duration: 10 seconds
Preparation Time: 3 seconds
4. Muscle Atrophy Test
Instruction: Squeeze the hand into a fist and hold.
Duration: 10 seconds
Preparation Time: 3 seconds
Future Enhancements
Integration with additional machine learning models for better prediction accuracy.
Support for more rehabilitation tasks and exercises.
