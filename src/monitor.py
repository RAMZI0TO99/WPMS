import cv2
import threading
import time
import queue
from collections import deque
import sys
import os
from ultralytics import YOLO

# Add parent dir to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker_state import StateMachine
import database

def video_capture_thread(video_source, frame_deque, deque_lock, stop_event):
    cap = cv2.VideoCapture(video_source)
    print(f"[VIDEO] Started capture from {video_source}")
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if isinstance(video_source, str) and os.path.isfile(video_source):
                print("[VIDEO] End of video file reached.")
                stop_event.set()
                break
            print("[VIDEO] Failed to read frame")
            time.sleep(0.1)
            continue
            
        with deque_lock:
            frame_deque.append(frame)
            
    cap.release()
    print("[VIDEO] Stopped capture")

def database_worker_thread(db_queue, stop_event):
    print("[DATABASE] Started worker thread")
    
    while not stop_event.is_set():
        try:
            event = db_queue.get(timeout=1.0)
        except queue.Empty:
            continue
            
        event_type = event[0]
        track_id = event[1]
        
        if event_type == 'ENTRY':
            role = event[2]
            timestamp = event[3]
            database.log_entry(track_id, role, timestamp)
        elif event_type == 'EXIT':
            timestamp = event[3]
            database.log_exit(track_id, timestamp)
            
        db_queue.task_done()
        
    print("[DATABASE] Stopped worker thread")

def inference_thread(frame_deque, deque_lock, db_queue, stop_event, model_path='yolov8n.pt', conf_thresh=0.25):
    print(f"[INFERENCE] Loading model {model_path}...")
    model = YOLO(model_path)
    state_machine = StateMachine(db_queue=db_queue, grace_period=5.0)
    
    print("[INFERENCE] Ready")
    
    while not stop_event.is_set():
        frame = None
        with deque_lock:
            if len(frame_deque) > 0:
                frame = frame_deque.pop()
                frame_deque.clear()
                
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Run YOLO with ByteTrack
        # agnostic_nms=True forces YOLO to keep only the highest confidence class for overlapping boxes
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=conf_thresh, agnostic_nms=True)
        result = results[0]
        
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                class_name = model.names[cls_id]
                
                # Check if it's the manager we want to track
                if 'manager' in class_name.lower():
                    role = class_name
                    state_machine.update_track(track_id, role)
                    
                    label = f"Track: {track_id} | {role}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # else:
                #     # Commented out to visually filter out workers as requested.
                #     label = f"{class_name}"
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
        state_machine.process_exits()
        
        # Resize frame to a manageable size for display
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Monitor", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            
    cv2.destroyAllWindows()
    print("[INFERENCE] Stopped thread")

def start_system(video_source=0, model_path='yolov8n.pt', conf_thresh=0.25):
    database.init_db()
    
    stop_event = threading.Event()
    db_queue = queue.Queue()
    frame_deque = deque(maxlen=2)
    deque_lock = threading.Lock()
    
    db_thread = threading.Thread(target=database_worker_thread, args=(db_queue, stop_event), daemon=True)
    video_thread = threading.Thread(target=video_capture_thread, args=(video_source, frame_deque, deque_lock, stop_event), daemon=True)
    
    db_thread.start()
    video_thread.start()
    
    try:
        inference_thread(frame_deque, deque_lock, db_queue, stop_event, model_path=model_path, conf_thresh=conf_thresh)
    except KeyboardInterrupt:
        print("\nStopping system...")
        stop_event.set()
        
    video_thread.join()
    db_queue.join()
    stop_event.set()
    db_thread.join()
    print("System fully shut down.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Vest Detection Monitor")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam, or path to mp4)")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO weights (use best.pt after training)")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold for YOLO detections")
    args = parser.parse_args()
    
    try:
        src = int(args.source)
    except ValueError:
        src = args.source
        
    start_system(video_source=src, model_path=args.model, conf_thresh=args.conf)
