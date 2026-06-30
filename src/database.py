import sqlite3
import time
import os

# Resolve the database path relative to this file's location so it always
# points to the project root regardless of the working directory at launch.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
DB_PATH = os.path.join(_PROJECT_ROOT, "workplace.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                camera_id TEXT NOT NULL DEFAULT '0',
                role TEXT NOT NULL,
                entry_time REAL NOT NULL,
                exit_time REAL
            )
        ''')
        
        # Check if camera_id exists, if not, add it for backwards compatibility
        cursor.execute("PRAGMA table_info(events)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'camera_id' not in columns:
            cursor.execute("ALTER TABLE events ADD COLUMN camera_id TEXT NOT NULL DEFAULT '0'")
            
        # CLEANUP: Close any active sessions left open from the previous run abruptly stopping
        cursor.execute("UPDATE events SET exit_time = entry_time WHERE exit_time IS NULL")
            
        conn.commit()

def log_entry(track_id, role="manager", current_time=None, camera_id="0"):
    """
    Handles ENTRY events with stateless session merging.
    If the last EXIT for this track_id on this camera was < 60 seconds ago, it resumes the session.
    """
    if current_time is None:
        current_time = time.time()
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check the most recent event for this track
        cursor.execute('''
            SELECT id, exit_time FROM events 
            WHERE track_id = ? AND camera_id = ?
            ORDER BY entry_time DESC LIMIT 1
        ''', (track_id, camera_id))
        row = cursor.fetchone()
        
        if row and row[1] is not None:
            last_event_id = row[0]
            last_exit_time = row[1]
            
            # Session Cooldown Logic: If they exited less than 60s ago, merge the session
            if (current_time - last_exit_time) < 60:
                cursor.execute('UPDATE events SET exit_time = NULL WHERE id = ?', (last_event_id,))
                conn.commit()
                return

        # Otherwise, insert a new entry
        cursor.execute('INSERT INTO events (track_id, role, entry_time, camera_id) VALUES (?, ?, ?, ?)',
                       (track_id, role, current_time, camera_id))
        conn.commit()

def log_exit(track_id, current_time=None, camera_id="0"):
    """
    Handles EXIT events by updating the most recent open session for this track on this camera.
    """
    if current_time is None:
        current_time = time.time()
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Find the most recent open session for this track
        cursor.execute('''
            SELECT id FROM events 
            WHERE track_id = ? AND camera_id = ? AND exit_time IS NULL 
            ORDER BY entry_time DESC LIMIT 1
        ''', (track_id, camera_id))
        row = cursor.fetchone()
        
        if row:
            event_id = row[0]
            cursor.execute('UPDATE events SET exit_time = ? WHERE id = ?', (current_time, event_id))
            conn.commit()
