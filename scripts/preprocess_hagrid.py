import zipfile
import sys
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import multiprocessing
import contextlib

DIMESION_SIZE = 8

def process_landmarks(results):
    """Compute 138 dimensional feature vector."""

    output_vector = np.zeros(138, dtype=np.float32)
    unnormalized_landmarks = {}

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label.lower()
            unnormalized_landmarks[hand_label] = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    anchor_hand_label = 'right' if 'right' in unnormalized_landmarks else 'left' if 'left' in unnormalized_landmarks else None

    if anchor_hand_label:
        anchor_wrist_pos = unnormalized_landmarks[anchor_hand_label][0]
        
        wrist_to_mcp_dist = np.linalg.norm(unnormalized_landmarks[anchor_hand_label][0] - unnormalized_landmarks[anchor_hand_label][9])
        scale_factor = wrist_to_mcp_dist if wrist_to_mcp_dist > 1e-5 else 0.1

        for hand_label, lm_list in unnormalized_landmarks.items():
            centered_pose = lm_list - anchor_wrist_pos
            normalized_pose = centered_pose / scale_factor
            flat_pose = normalized_pose.flatten()
            
            if hand_label == 'left':
                output_vector[0:63] = flat_pose
            elif hand_label == 'right':
                output_vector[69:132] = flat_pose
                
    return output_vector

def consumer(task_queue, results_queue, lock, output_dir, log_file_path):
    """Process images from queue and save as .npy."""

    with open(os.devnull, 'w') as f, contextlib.redirect_stderr(f):
        detector = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            
    while True:
        image_entry = task_queue.get()
        if image_entry is None: break
        
        image_path_in_zip, image_data = image_entry
        try:
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None: raise ValueError('Failed to decode image')
            
            results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks: raise ValueError('No hands detected')
            
            feature_vector = process_landmarks(results)

            static_motion_clip = np.tile(feature_vector, (DIMESION_SIZE, 1))

            base_name_with_dirs, _ = os.path.splitext(image_path_in_zip)
            output_npy_path = os.path.join(output_dir, f"{base_name_with_dirs}.npy")
            
            os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
            np.save(output_npy_path, static_motion_clip)

        except Exception as e:
            log_message = f"{image_path_in_zip} (Reason: {e})\n"
            with lock:
                with open(log_file_path, 'a') as log_file: log_file.write(log_message)
        finally:
            results_queue.put(1)

def producer(task_queue, zip_file_path, file_list):
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        for image_path in file_list:
            task_queue.put((image_path, zipf.read(image_path)))

def convert_all(zip_file_path, output_dir, log_file_path="failed_images.txt"):
    if not os.path.exists(zip_file_path):
        print(f"Error: Zip file not found at '{zip_file_path}'")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    with open(log_file_path, 'a') as f: f.write(f"\n--- Starting new zip-to-npy run ---\n")
    
    print("Scanning zip file and checking for previously processed files...")
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        all_files_in_zip = [f for f in zipf.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('__MACOSX/')]
        
    files_to_process = []
    for image_path in all_files_in_zip:
        base_name_with_dirs, _ = os.path.splitext(image_path)
        expected_output_path = os.path.join(output_dir, f"{base_name_with_dirs}.npy")
        if not os.path.exists(expected_output_path):
            files_to_process.append(image_path)
            
    total_files_in_zip = len(all_files_in_zip)
    remaining_files = len(files_to_process)
    print(f"Found {total_files_in_zip} total images. {remaining_files} remaining to process.")
    if not files_to_process:
        print("All images have already been processed.")
        return

    num_consumers = multiprocessing.cpu_count()
    print(f"Starting 1 producer and {num_consumers} consumer processes...")

    task_queue = multiprocessing.Queue(maxsize=num_consumers * 2)
    results_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()

    producer_process = multiprocessing.Process(target=producer, args=(task_queue, zip_file_path, files_to_process))
    producer_process.start()

    consumer_processes = []
    for _ in range(num_consumers):
        p = multiprocessing.Process(target=consumer, args=(task_queue, results_queue, lock, output_dir, log_file_path))
        p.start()
        consumer_processes.append(p)

    for _ in tqdm(range(remaining_files), desc="Processing images"):
        results_queue.get()

    print("\nAll images have been processed. Shutting down workers...")
    producer_process.join()
    for _ in range(num_consumers):
        task_queue.put(None)
    for p in consumer_processes:
        p.join()
    print("All processes have completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    zip_file_to_process = input("Enter the path to the hagrid zip file: ")
    output = input("Enter the name for the output directory: ")
    convert_all(zip_file_to_process, output)