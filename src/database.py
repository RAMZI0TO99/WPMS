import sqlite3
import time
import os

DB_PATH = "workplace.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if os.path.dirname(DB_PATH) else None
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                entry_time REAL NOT NULL,
                exit_time REAL
            )
        ''')
        conn.commit()

def log_entry(track_id, role="manager", current_time=None):
    """
    Handles ENTRY events with stateless session merging.
    If the last EXIT for this track_id was < 60 seconds ago, it resumes the session.
    """
    if current_time is None:
        current_time = time.time()
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Check the most recent event for this track
        cursor.execute('''
            SELECT id, exit_time FROM events 
            WHERE track_id = ? 
            ORDER BY entry_time DESC LIMIT 1
        ''', (track_id,))
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
        cursor.execute('INSERT INTO events (track_id, role, entry_time) VALUES (?, ?, ?)',
                       (track_id, role, current_time))
        conn.commit()

def log_exit(track_id, current_time=None):
    """
    Handles EXIT events by updating the most recent open session for this track.
    """
    if current_time is None:
        current_time = time.time()
        
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Find the most recent open session for this track
        cursor.execute('''
            SELECT id FROM events 
            WHERE track_id = ? AND exit_time IS NULL 
            ORDER BY entry_time DESC LIMIT 1
        ''', (track_id,))
        row = cursor.fetchone()
        
        if row:
            event_id = row[0]
            cursor.execute('UPDATE events SET exit_time = ? WHERE id = ?', (current_time, event_id))
            conn.commit()
