import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import contextlib
from collections import deque

def process_single_video(video_dir, detector):
    """Process all frames in a video directory and return numpy array [num_frames, 138]."""

    # Sort by the integer value of the filename to ensure correct order
    image_files = sorted(list(Path(video_dir).glob('*.jpg')), key=lambda p: int(p.stem))
    if not image_files: return None

    raw_frame_data = []
    for img_path in image_files:
        frame_landmarks = {}
        image = cv2.imread(str(img_path))
        if image is None:
            raw_frame_data.append(frame_landmarks); continue
        
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
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

def consumer(task_queue, results_queue, lock, output_dir, log_file_path):
    """Process video directories and save .npy files."""

    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        detector = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        video_dir_str = task_queue.get()
        if video_dir_str is None: break
        
        video_dir = Path(video_dir_str)
        try:
            full_video_features = process_single_video(video_dir, detector)
            if full_video_features is None: raise ValueError("No images found in directory")

            output_path = Path(output_dir) / f"{video_dir.name}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, full_video_features)
        except Exception as e:
            log_message = f"{video_dir_str} (Reason: {e})\n"
            with lock:
                with open(log_file_path, 'a') as log_file: log_file.write(log_message)
        finally:
            results_queue.put(1)

def producer(task_queue, task_list):
    """Add video directory paths to queue."""

    for task in task_list:
        task_queue.put(str(task))

def main():
    """Preprocess Jester dataset."""

    jester_dir = input("Enter the path to the unzipped Jester image frames directory: ")
    output_dir = input("Enter the path for the output .npy directory: ")
    log_file_path = "failed_jester_videos.txt"

    if not Path(jester_dir).is_dir():
        print(f"Error: Input directory not found at '{jester_dir}'"); return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(log_file_path, 'a') as f: f.write(f"\n--- Starting new Jester run ---\n")

    print("Scanning for all video folders in the dataset...")
    all_video_dirs = [d for d in Path(jester_dir).iterdir() if d.is_dir() and d.name.isdigit()]

    print("Scanning for previously processed videos to resume...")
    tasks_to_process = []
    for video_dir in all_video_dirs:
        expected_output_path = Path(output_dir) / f"{video_dir.name}.npy"
        if not expected_output_path.exists():
            tasks_to_process.append(video_dir)
            
    total_videos = len(all_video_dirs)
    remaining_videos = len(tasks_to_process)
    print(f"Found {total_videos} total videos. {remaining_videos} remaining to process.")
    if not tasks_to_process:
        print("All videos have already been processed."); return

    num_consumers = multiprocessing.cpu_count()
    print(f"Starting 1 producer and {num_consumers} consumer processes...")
    task_queue = multiprocessing.Queue(maxsize=num_consumers * 2)
    results_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    producer_process = multiprocessing.Process(target=producer, args=(task_queue, tasks_to_process))
    producer_process.start()

    consumer_processes = []
    for _ in range(num_consumers):
        p = multiprocessing.Process(target=consumer, args=(task_queue, results_queue, lock, output_dir, log_file_path))
        p.start()
        consumer_processes.append(p)

    for _ in tqdm(range(remaining_videos), desc="Processing videos"):
        results_queue.get()

    print("\nAll videos have been processed. Shutting down workers...")
    producer_process.join()
    for _ in range(num_consumers):
        task_queue.put(None)
    for p in consumer_processes:
        p.join()
    
    print("All processes have completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()