import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
from collections import deque

# Config
OUTPUT_DIR = "processed/custom_video_data"
WINDOW_SIZE = 8 # Minimum frames to save
MAX_HANDS = 2

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_feature_vector(results, history):
    """Calculates the 138-dim feature vector (Left+Right Pose, Vel, Acc)."""
    raw_landmarks = {}
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label.lower()
            raw_landmarks[label] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # Get anchor point
    anchor = 'right' if 'right' in raw_landmarks else 'left' if 'left' in raw_landmarks else None
    
    # Calculate Normalization
    anchor_pos = np.zeros(3)
    scale = 1.0
    if anchor:
        anchor_pos = raw_landmarks[anchor][0] # Wrist
        # Scale based on wrist-to-middle-finger-mcp distance
        scale = np.linalg.norm(raw_landmarks[anchor][0] - raw_landmarks[anchor][9])
        if scale < 1e-5: scale = 1.0

    vector = np.zeros(138, dtype=np.float32)
    
    for label in ['left', 'right']:
        if label in raw_landmarks:
            history[label]['pos'].append(raw_landmarks[label][0]) # Track wrist
            
            norm_pose = (raw_landmarks[label] - anchor_pos) / scale
            
            vel = np.zeros(3)
            if len(history[label]['pos']) >= 2:
                vel = history[label]['pos'][-1] - history[label]['pos'][-2]
            history[label]['vel'].append(vel)

            acc = np.zeros(3)
            if len(history[label]['vel']) >= 2:
                acc = history[label]['vel'][-1] - history[label]['vel'][-2]

            start = 0 if label == 'left' else 69
            vector[start:start+63] = norm_pose.flatten()
            vector[start+63:start+66] = vel
            vector[start+66:start+69] = acc
        else:
            history[label]['pos'].clear()
            history[label]['vel'].clear()

    return vector

def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=0.5)
    
    history = {
        'left': {'pos': deque(maxlen=5), 'vel': deque(maxlen=5)},
        'right': {'pos': deque(maxlen=5), 'vel': deque(maxlen=5)}
    }

    is_recording = False
    recorded_frames = []

    print(f"Saving to: {OUTPUT_DIR}")
    print("Controls: 'r' to Record, 's' to Stop/Save, 'q' to Quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw
        if results.multi_hand_landmarks:
            for lh in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lh, mp_hands.HAND_CONNECTIONS)

        # Record
        if is_recording:
            feat_vec = get_feature_vector(results, history)
            recorded_frames.append(feat_vec)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Red dot
            cv2.putText(frame, f"REC: {len(recorded_frames)}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            is_recording = True
            recorded_frames = []
            print(">>> Recording started...")
        elif key == ord('s'):
            is_recording = False
            if len(recorded_frames) > WINDOW_SIZE:
                path = os.path.join(OUTPUT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
                np.save(path, np.array(recorded_frames, dtype=np.float32))
                print(f"SAVED: {path}")
            else:
                print("Discarded (too short)")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()