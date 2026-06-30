import os
# ── Suppress TensorFlow / absl-py log spam ────────────────────────────────────
# Must be set BEFORE importing tensorflow, deepface, or ultralytics so the
# env vars are seen at library init time.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # hide TF C++ info/warn
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")      # suppress oneDNN notice
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")         # suppress absl InitializeLog
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")         # hide gRPC noise

import cv2
import threading
import time
import queue
from collections import deque
import sys
import concurrent.futures
import torch
from ultralytics import YOLO

# ─────────────────────────────────────────────
# Project root: parent of this script's directory (src/)
# Used to resolve paths absolutely regardless of CWD.
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_MANAGERS_DIR = os.path.join(PROJECT_ROOT, "known_managers")

# Auto-detect best compute device (CUDA GPU > CPU)
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"[SYSTEM] GPU detected: {torch.cuda.get_device_name(0)} — Running on CUDA.")
elif torch.version.cuda is None:
    DEVICE = 'cpu'
    print("[SYSTEM] ⚠️  PyTorch was installed WITHOUT CUDA support (CPU-only wheel).")
    print("[SYSTEM]    Run: uv sync --reinstall-package torch torchvision")
    print("[SYSTEM]    to reinstall with the CUDA 12.4 build (see pyproject.toml).")
else:
    DEVICE = 'cpu'
    print(f"[SYSTEM] PyTorch built with CUDA {torch.version.cuda} but no compatible GPU found.")
    print("[SYSTEM]    Check your NVIDIA driver with: nvidia-smi")

# Add the project root at the FRONT of sys.path so local modules (database,
# face_id, tracker_state) always take precedence over any same-named
# installed packages.
sys.path.insert(0, PROJECT_ROOT)

from tracker_state import StateMachine
import database
from face_id import FaceIdentifier


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
            camera_id = event[4]
            database.log_entry(track_id, role, timestamp, camera_id)
        elif event_type == 'EXIT':
            timestamp = event[3]
            camera_id = event[4]
            database.log_exit(track_id, timestamp, camera_id)
            
        db_queue.task_done()
        
    print("[DATABASE] Stopped worker thread")


def _crop_frame(frame, x1, y1, x2, y2):
    """Safely crop a frame region and return a copy."""
    y1_c = max(0, y1)
    y2_c = min(frame.shape[0], y2)
    x1_c = max(0, x1)
    x2_c = min(frame.shape[1], x2)
    return frame[y1_c:y2_c, x1_c:x2_c].copy()


def inference_thread(frame_deque, deque_lock, db_queue, display_queue,
                     stop_event, model_path=None, conf_thresh=0.25,
                     camera_id="0", min_frames=3):
    # Resolve model_path against PROJECT_ROOT when the caller passes the bare
    # filename default, so the weights are found regardless of CWD.
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'yolov8n.pt')
    print(f"[INFERENCE] Loading model {model_path} for camera {camera_id}...")
    model = YOLO(model_path)
    state_machine = StateMachine(db_queue=db_queue, camera_id=camera_id,
                                 grace_period=5.0, min_frames=min_frames)
    
    # Pass the absolute known_managers path so it always resolves correctly
    face_identifier = FaceIdentifier(known_faces_dir=KNOWN_MANAGERS_DIR)
    
    # max_workers=2 allows scanning two people simultaneously per camera
    face_id_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    face_id_futures = {}  # track_id -> Future

    # track_id -> role string:
    #   "Worker"            – confirmed non-manager
    #   "Manager (Unknown)" – YOLO confirmed manager, name not yet identified
    #   "Manager (Name)"    – YOLO confirmed + face identified
    #   "Scanning..."       – generic model: pending face ID to determine role
    known_tracks = {}
    known_track_retries = {}
    
    print(f"[INFERENCE] Ready. Camera {camera_id} | Device: {DEVICE}")

    while not stop_event.is_set():
        frame = None
        with deque_lock:
            if len(frame_deque) > 0:
                frame = frame_deque.pop()
                frame_deque.clear()
                
        if frame is None:
            time.sleep(0.01)
            continue
            
        # Run YOLO with ByteTrack on best available device
        # agnostic_nms=True: keep only highest-confidence class for overlapping boxes
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=conf_thresh,
            agnostic_nms=True,
            device=DEVICE
        )
        result = results[0]
        
        if result.boxes.id is not None:
            boxes      = result.boxes.xyxy.cpu().numpy()
            track_ids  = result.boxes.id.cpu().numpy().astype(int)
            class_ids  = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[cls_id]
                
                # ── Determine role from YOLO class name ─────────────────────
                # Your custom model should produce class names like "manager" or "worker".
                # If using the generic yolov8n.pt, all detections are "person" and
                # face ID is used to distinguish managers from workers.
                MANAGER_KEYWORDS = ["manager", "vest", "supervisor", "foreman"]
                WORKER_KEYWORDS  = ["worker", "employee", "staff", "laborer"]
                
                is_yolo_manager = any(kw in class_name.lower() for kw in MANAGER_KEYWORDS)
                is_yolo_worker  = any(kw in class_name.lower() for kw in WORKER_KEYWORDS)
                # Generic model (e.g. yolov8n.pt) only knows "person":
                is_generic_person = not is_yolo_manager and not is_yolo_worker
                
                # ── First time we see this track ─────────────────────────────
                if track_id not in known_tracks:
                    known_track_retries[track_id] = 0
                    
                    if is_yolo_worker:
                        # Custom model confirmed worker → skip face scan
                        known_tracks[track_id] = "Worker"
                        
                    elif is_yolo_manager:
                        # Custom model confirmed manager → show immediately, scan for name
                        known_tracks[track_id] = "Manager (Unknown)"
                        print(f"[DETECT] Manager (class='{class_name}') detected on Camera {camera_id} Track {track_id}. Scanning for name...")
                        crop = _crop_frame(frame, x1, y1, x2, y2)
                        if crop.size > 0 and track_id not in face_id_futures:
                            face_id_futures[track_id] = face_id_executor.submit(face_identifier.identify_face, crop)
                            
                    else:
                        # Generic model: role unknown → scan face to identify
                        known_tracks[track_id] = "Scanning..."
                        crop = _crop_frame(frame, x1, y1, x2, y2)
                        if crop.size > 0 and track_id not in face_id_futures:
                            face_id_futures[track_id] = face_id_executor.submit(face_identifier.identify_face, crop)
                
                # ── Check pending face ID futures ────────────────────────────
                elif known_tracks[track_id] in ("Scanning...", "Manager (Unknown)"):
                    still_scanning = known_tracks[track_id]
                    
                    if track_id in face_id_futures and face_id_futures[track_id].done():
                        try:
                            name = face_id_futures[track_id].result()
                        except Exception:
                            name = None
                        del face_id_futures[track_id]
                        
                        if name:
                            # Successfully identified!
                            known_tracks[track_id] = f"Manager ({name})"
                            print(f"[FACE ID] ✓ Identified Manager '{name}' (Track {track_id}, Camera {camera_id})")
                        else:
                            # No match this attempt
                            known_track_retries[track_id] += 1
                            
                            if known_track_retries[track_id] < 10:
                                # Retry with a fresh crop (person may have turned to face camera)
                                crop = _crop_frame(frame, x1, y1, x2, y2)
                                if crop.size > 0:
                                    face_id_futures[track_id] = face_id_executor.submit(face_identifier.identify_face, crop)
                            else:
                                # All 10 retries exhausted
                                if still_scanning == "Manager (Unknown)":
                                    # YOLO confirmed manager but name unknown → keep showing as Manager
                                    print(f"[FACE ID] Manager (Track {track_id}) name unknown after 10 tries. Staying as 'Manager (Unknown)'.")
                                else:
                                    # Generic model: not recognized → classify as Worker
                                    known_tracks[track_id] = "Worker"
                
                # ── Render the bounding box ──────────────────────────────────
                role = known_tracks.get(track_id)
                
                if role and "Manager" in role:
                    # Confirmed or YOLO-detected manager — red box
                    state_machine.update_track(track_id, role)
                    label = f"ID:{track_id} | {role}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif role == "Scanning...":
                    # Generic model: show brief scanning indicator on first attempt only
                    if known_track_retries.get(track_id, 0) < 2:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                        cv2.putText(frame, "Scanning...", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                # Workers are intentionally not rendered for a clean view
                    
        # Collect expired track IDs so we can clean up per-track state and
        # prevent the known_tracks / face_id_futures dicts growing forever.
        expired_ids = state_machine.process_exits()
        for tid in expired_ids:
            known_tracks.pop(tid, None)
            known_track_retries.pop(tid, None)
            future = face_id_futures.pop(tid, None)
            if future is not None and not future.done():
                future.cancel()
        
        # Push the annotated frame to the main-thread display queue instead of
        # calling cv2.imshow here. cv2 GUI calls MUST run on the main thread on
        # Windows and macOS or they crash / silently do nothing.
        display_frame = cv2.resize(frame, (1280, 720))
        try:
            display_queue.put_nowait((camera_id, display_frame))
        except Exception:
            pass  # drop frame if queue is full — display will catch up
            
    # Cleanup
    face_id_executor.shutdown(wait=False)
    print("[INFERENCE] Stopped thread")


def start_system(video_sources=None, model_path=None, conf_thresh=0.25, min_frames=3):
    # Guard against the mutable-default-argument pitfall.
    if video_sources is None:
        video_sources = [0]
    # Resolve a bare filename against the project root so the model is always
    # found no matter which directory the user launches from.
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, 'yolov8n.pt')
    database.init_db()
    
    stop_event = threading.Event()
    db_queue = queue.Queue()
    # display_queue carries (camera_id, frame) tuples from inference threads
    # to the main thread, which is the only thread allowed to call cv2.imshow.
    display_queue = queue.Queue(maxsize=4)
    
    db_thread = threading.Thread(
        target=database_worker_thread,
        args=(db_queue, stop_event),
        daemon=True
    )
    db_thread.start()
    
    threads = []
    
    for idx, source in enumerate(video_sources):
        camera_id = str(source)
        frame_deque = deque(maxlen=2)
        deque_lock = threading.Lock()
        
        video_thread = threading.Thread(
            target=video_capture_thread,
            args=(source, frame_deque, deque_lock, stop_event),
            daemon=True
        )
        inf_thread = threading.Thread(
            target=inference_thread,
            args=(frame_deque, deque_lock, db_queue, display_queue,
                  stop_event, model_path, conf_thresh, camera_id, min_frames),
            daemon=True
        )
        
        video_thread.start()
        inf_thread.start()
        threads.extend([video_thread, inf_thread])
    
    # ── Main-thread display loop ─────────────────────────────────────────────
    # cv2.imshow / cv2.waitKey MUST be called from the main thread on Windows
    # and macOS. Inference threads push annotated frames here via display_queue.
    try:
        while not stop_event.is_set():
            try:
                camera_id_disp, frame_disp = display_queue.get(timeout=0.05)
                cv2.imshow(f"Monitor - {camera_id_disp}", frame_disp)
            except queue.Empty:
                pass
            # waitKey keeps the OpenCV window responsive; 'q' signals all threads.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
    except KeyboardInterrupt:
        print("\nStopping system...")
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
        
    for t in threads:
        t.join()
        
    db_queue.join()
    stop_event.set()
    db_thread.join()
    print("System fully shut down.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Workplace Monitor")
    parser.add_argument("--source", nargs="+", default=[0],
                        help="Video sources (0 for webcam, or paths to mp4s)")
    # Default to the best custom-trained model in the project.
    # This model knows "manager" and "worker" classes unlike the generic yolov8n.
    _default_model = os.path.join(
        PROJECT_ROOT, "runs", "detect", "train-4", "weights", "best.pt"
    )
    if not os.path.exists(_default_model):
        # Fall back to the pretrained generic model if no custom model exists yet.
        _default_model = os.path.join(PROJECT_ROOT, "yolov8n.pt")
        print(f"[WARNING] Custom model not found, falling back to: {_default_model}")
        print("[WARNING] Generic yolov8n only knows 'person' — train a custom model for")
        print("[WARNING] manager/worker detection: uv run python src/training/train.py --data dataset/data.yaml")

    parser.add_argument("--model", default=_default_model,
                        help="Path to YOLO weights (default: custom best.pt from train-4)")
    parser.add_argument("--conf", type=float, default=0.75,
                        help="Confidence threshold for YOLO detections (default: 0.75)")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="Min consecutive frames a track must appear before logging ENTRY (default: 3, increase to reduce false positives)")
    args = parser.parse_args()
    
    sources = []
    for s in args.source:
        try:
            sources.append(int(s))
        except ValueError:
            sources.append(s)
        
    start_system(video_sources=sources, model_path=args.model,
                 conf_thresh=args.conf, min_frames=args.min_frames)
