import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

VIS_SCALE = 200
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
WHITE_COLOR = (255, 255, 255)

def draw_normalized_skeleton(frame, normalized_landmarks, color, offset_x):
    """Draws a normalized hand skeleton with a screen offset."""
    if normalized_landmarks is None:
        return

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_pt = normalized_landmarks[start_idx]
        end_pt = normalized_landmarks[end_idx]

        start_x = int(start_pt[0] * VIS_SCALE + center_x + offset_x)
        start_y = int(start_pt[1] * VIS_SCALE + center_y)
        end_x = int(end_pt[0] * VIS_SCALE + center_x + offset_x)
        end_y = int(end_pt[1] * VIS_SCALE + center_y)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)

def run_visualization():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # History for calculating velocity/acceleration
    history_buffers = {
        'left': deque(maxlen=3),
        'right': deque(maxlen=3)
    }

    # Higher confidence helps stabilize the two-hand tracking
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        normalized_poses = {'left': None, 'right': None}
        velocities = {'left': np.zeros(3), 'right': np.zeros(3)}
        accelerations = {'left': np.zeros(3), 'right': np.zeros(3)}

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        unnormalized_landmarks = {}
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # MediaPipe doesn't guarantee order, so we check the label
                hand_label = results.multi_handedness[i].classification[0].label.lower()
                unnormalized_landmarks[hand_label] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # Draw live overlay
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=GREEN_COLOR, thickness=2)
                )

        # Calculate Shared Space Normalization
        # We center based on the midpoint of whatever hands are visible
        midpoint = np.zeros(3)
        if 'left' in unnormalized_landmarks and 'right' in unnormalized_landmarks:
            midpoint = (unnormalized_landmarks['left'][0] + unnormalized_landmarks['right'][0]) / 2.0
        elif 'left' in unnormalized_landmarks:
            midpoint = unnormalized_landmarks['left'][0]
        elif 'right' in unnormalized_landmarks:
            midpoint = unnormalized_landmarks['right'][0]

        if unnormalized_landmarks:
            all_landmarks_centered = []
            temp_poses = {}
            
            for hand_label, lm_list in unnormalized_landmarks.items():
                centered_pose = lm_list - midpoint
                temp_poses[hand_label] = centered_pose
                all_landmarks_centered.extend(centered_pose)
            
            # Normalize by the max distance in the set to keep relative scale
            max_dist = np.max(np.linalg.norm(all_landmarks_centered, axis=1))
            if max_dist > 0:
                for hand_label, centered_pose in temp_poses.items():
                    normalized_poses[hand_label] = centered_pose / max_dist

        # Update derivatives (Motion streams)
        for hand_label in ['left', 'right']:
            if hand_label in unnormalized_landmarks:
                wrist_pos = unnormalized_landmarks[hand_label][0]
                history_buffers[hand_label].append(wrist_pos)

                if len(history_buffers[hand_label]) >= 2:
                    velocities[hand_label] = history_buffers[hand_label][-1] - history_buffers[hand_label][-2]
                if len(history_buffers[hand_label]) >= 3:
                    prev_velocity = history_buffers[hand_label][-2] - history_buffers[hand_label][-3]
                    accelerations[hand_label] = velocities[hand_label] - prev_velocity
            else:
                history_buffers[hand_label].clear()

        # Visualize
        # Draw normalized skeletons with offsets to center them side-by-side
        draw_normalized_skeleton(frame, normalized_poses['left'], RED_COLOR, offset_x=-VIS_SCALE/2)
        draw_normalized_skeleton(frame, normalized_poses['right'], RED_COLOR, offset_x=VIS_SCALE/2)
        
        # UI Overlay
        cv2.rectangle(frame, (0,0), (w, 180), (0,0,0), -1)
        cv2.putText(frame, "LIVE HANDS (Green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN_COLOR, 2)
        cv2.putText(frame, "NORMALIZED POSE (Red, Centered)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        
        l_vel, l_acc = velocities['left'], accelerations['left']
        cv2.putText(frame, f"L. Vel: [{l_vel[0]:.2f}, {l_vel[1]:.2f}, {l_vel[2]:.2f}]", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1)
        cv2.putText(frame, f"L. Acc: [{l_acc[0]:.2f}, {l_acc[1]:.2f}, {l_acc[2]:.2f}]", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1)

        r_vel, r_acc = velocities['right'], accelerations['right']
        cv2.putText(frame, f"R. Vel: [{r_vel[0]:.2f}, {r_vel[1]:.2f}, {r_vel[2]:.2f}]", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1)
        cv2.putText(frame, f"R. Acc: [{r_acc[0]:.2f}, {r_acc[1]:.2f}, {r_acc[2]:.2f}]", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE_COLOR, 1)

        cv2.imshow('Two-Handed Feature Visualization', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_visualization()