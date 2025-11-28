import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from model.vqvae import VQVAE

MODEL_PATH = 'best_gesture_model.pth'
WINDOW_SIZE = 8
INPUT_DIM = 138 * WINDOW_SIZE
EMBED_DIM = 128
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25

# Visualize
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def draw_normalized_skeleton(frame, norm_pose, anchor_pos, scale, color):
    h, w, c = frame.shape
    landmarks = norm_pose.reshape(21, 3)
    screen_points = []
    for lm in landmarks:
        pt = (lm * scale) + anchor_pos
        px = int(pt[0] * w); py = int(pt[1] * h)
        screen_points.append((px, py))
        
    for start, end in CONNECTIONS:
        if start < 21 and end < 21:
            cv2.line(frame, screen_points[start], screen_points[end], color, 2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Load Model
    model = VQVAE(INPUT_DIM, EMBED_DIM, NUM_EMBEDDINGS, COMMITMENT_COST).to(device)
    try: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except: return print("Model not found!")
    model.eval()

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    
    history = {'left': {'pos': deque(maxlen=5), 'vel': deque(maxlen=5)}, 
               'right': {'pos': deque(maxlen=5), 'vel': deque(maxlen=5)}}
    model_buffer = deque(maxlen=WINDOW_SIZE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raw_landmarks = {}
        if results.multi_hand_landmarks:
            for i, hl in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hl, CONNECTIONS, mp_drawing.DrawingSpec(color=(0,255,0)))
                label = results.multi_handedness[i].classification[0].label.lower()
                raw_landmarks[label] = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])

        # Feature Logic
        anchor = 'right' if 'right' in raw_landmarks else 'left' if 'left' in raw_landmarks else None
        anchor_pos = np.zeros(3); scale = 1.0
        if anchor:
            anchor_pos = raw_landmarks[anchor][0]
            scale = np.linalg.norm(raw_landmarks[anchor][0] - raw_landmarks[anchor][9])
            if scale < 1e-5: scale = 1.0

        vec = np.zeros(138, dtype=np.float32)
        for label in ['left', 'right']:
            if label in raw_landmarks:
                history[label]['pos'].append(raw_landmarks[label][0])
                norm_pose = (raw_landmarks[label] - anchor_pos) / scale
                vel = np.zeros(3); acc = np.zeros(3)
                if len(history[label]['pos']) >= 2:
                    vel = history[label]['pos'][-1] - history[label]['pos'][-2]
                history[label]['vel'].append(vel)
                if len(history[label]['vel']) >= 2:
                    acc = history[label]['vel'][-1] - history[label]['vel'][-2]
                
                s = 0 if label == 'left' else 69
                vec[s:s+63] = norm_pose.flatten()
                vec[s+63:s+66] = vel
                vec[s+66:s+69] = acc
            else:
                history[label]['pos'].clear(); history[label]['vel'].clear()
        
        model_buffer.append(vec)

        if len(model_buffer) == WINDOW_SIZE:
            inp = torch.from_numpy(np.array(model_buffer).flatten()).float().unsqueeze(0).to(device)
            with torch.no_grad():
                recon, _, indices = model.get_tokens_and_reconstructions(inp)
                token = indices.view(-1)[0].item()
            
            last_frame = recon.cpu().numpy().reshape(WINDOW_SIZE, 138)[-1]
            if anchor: 
                draw_normalized_skeleton(frame, last_frame[0:63], anchor_pos, scale, (0,0,255))
                draw_normalized_skeleton(frame, last_frame[69:132], anchor_pos, scale, (0,0,255))
                cv2.putText(frame, f"Token: {token}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.imshow("Gesture Visualizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()