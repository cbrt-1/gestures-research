import os
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import multiprocessing
import contextlib
from collections import deque

def process_single_video(cap, detector):
    """Process all frames from VideoCapture and return numpy array [num_frames, 138]."""

    raw_frame_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_landmarks = {}
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[i].classification[0].label.lower()
                frame_landmarks[hand_label] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        raw_frame_data.append(frame_landmarks)

    final_feature_vectors = []
    history_buffers = {'left': deque(maxlen=3), 'right': deque(maxlen=3)}

    for i in range(len(raw_frame_data)):
        unnormalized_landmarks = raw_frame_data[i]
        feature_vector = np.zeros(138, dtype=np.float32)

        anchor_hand_label = 'right' if 'right' in unnormalized_landmarks else 'left' if 'left' in unnormalized_landmarks else None
        if anchor_hand_label:
            anchor_wrist_pos = unnormalized_landmarks[anchor_hand_label][0]
            wrist_to_mcp_dist = np.linalg.norm(unnormalized_landmarks[anchor_hand_label][0] - unnormalized_landmarks[anchor_hand_label][9])
            scale_factor = wrist_to_mcp_dist if wrist_to_mcp_dist > 1e-5 else 0.1

            for hand_label, lm_list in unnormalized_landmarks.items():
                centered_pose = lm_list - anchor_wrist_pos
                normalized_pose = centered_pose / scale_factor
                if hand_label == 'left': feature_vector[0:63] = normalized_pose.flatten()
                elif hand_label == 'right': feature_vector[69:132] = normalized_pose.flatten()

        for hand_label in ['left', 'right']:
            if hand_label in unnormalized_landmarks:
                wrist_pos = unnormalized_landmarks[hand_label][0]
                history_buffers[hand_label].append(wrist_pos)
                velocity = np.zeros(3)
                acceleration = np.zeros(3)
                
                if len(history_buffers[hand_label]) >= 2:
                    velocity = history_buffers[hand_label][-1] - history_buffers[hand_label][-2]
                    
                if len(history_buffers[hand_label]) >= 3:
                    prev_velocity = history_buffers[hand_label][-2] - history_buffers[hand_label][-3]
                    acceleration = velocity - prev_velocity
                
                if hand_label == 'left':
                    feature_vector[63:66] = velocity
                    feature_vector[66:69] = acceleration
                elif hand_label == 'right':
                    feature_vector[132:135] = velocity
                    feature_vector[135:138] = acceleration
            else: history_buffers[hand_label].clear()

        final_feature_vectors.append(feature_vector)
    
    return np.array(final_feature_vectors, dtype=np.float32)


def consumer(task_queue, results_queue, output_dir, log_file_path, lock):
    """Process video files and save landmarks."""

    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
        )

    while True:
        video_path_str = task_queue.get()
        if video_path_str is None: break
        
        video_path = Path(video_path_str)
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened(): raise ValueError("Could not open video file")

            full_video_features = process_single_video(cap, hands)
            cap.release()

            if full_video_features is None or len(full_video_features) == 0:
                raise ValueError("No landmarks were extracted from the video")

            output_file = output_dir / (video_path.stem + ".npy")
            np.save(output_file, full_video_features)

        except Exception as e:
            with lock:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{video_path_str} (Reason: {e})\n")
        finally:
            results_queue.put(1)

def producer(task_queue, video_files):
    """Add video file paths to queue."""
    for video_file in video_files:
        task_queue.put(str(video_file))

def main():
    workspace_dir = Path(os.getcwd())
    video_dir = workspace_dir / "data" / "wlasl" / "videos"
    output_dir = workspace_dir / "data" / "processed" / "wlasl_landmarks_138"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    log_file_path = workspace_dir / "failed_wlasl_videos.txt"
    with open(log_file_path, 'a') as f:
        f.write(f"\n--- Starting new WLASL run ---\n")

    print(f"Scanning for video files in: {video_dir}")
    all_video_files = list(video_dir.glob("*.mp4"))

    if not all_video_files:
        print(f"No video files found. Please check the path."); return

    files_to_process = []
    for video_path in all_video_files:
        expected_output_path = output_dir / f"{video_path.stem}.npy"
        if not expected_output_path.exists():
            files_to_process.append(video_path)

    total_files = len(all_video_files)
    remaining_files = len(files_to_process)
    print(f"Found {total_files} total videos. {remaining_files} remaining to process.")
    if not files_to_process:
        print("All videos have already been processed."); return

    num_consumers = multiprocessing.cpu_count()
    print(f"Starting 1 producer and {num_consumers} consumer processes...")

    task_queue = multiprocessing.Queue(maxsize=num_consumers * 2)
    results_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    producer_process = multiprocessing.Process(target=producer, args=(task_queue, files_to_process))
    producer_process.start()

    consumer_processes = []
    for _ in range(num_consumers):
        p = multiprocessing.Process(target=consumer, args=(task_queue, results_queue, output_dir, log_file_path, lock))
        p.start()
        consumer_processes.append(p)

    for _ in tqdm(range(remaining_files), desc="Processing videos"):
        results_queue.get()

    print("\nAll videos have been processed. Shutting down workers...")
    producer_process.join()
    for _ in range(num_consumers):
        task_queue.put(None)
    for p in consumer_processes:
        p.join()
    
    print("All processes have completed.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()