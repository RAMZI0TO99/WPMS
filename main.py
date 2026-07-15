import os
import sys

from src.monitor import start_system

def main():
    videos_dir = "videos"
    
    if not os.path.exists(videos_dir):
        print(f"Error: '{videos_dir}' directory not found.")
        return

    # List all video files in the videos directory
    videos = [f for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not videos:
        print(f"No videos found in the '{videos_dir}' directory.")
        return

    print("--- Workplace Monitoring System ---")
    print("Available video sources:")
    for i, video in enumerate(videos):
        print(f"{i + 1}. {video}")
    
    # Add webcam as an option
    webcam_option = len(videos) + 1
    print(f"{webcam_option}. Webcam (Live)")

    try:
        choice = input(f"\nEnter the number of the video you want to test on (1-{webcam_option}): ")
        choice = int(choice)
        
        if 1 <= choice <= len(videos):
            selected_source = os.path.join(videos_dir, videos[choice - 1])
            # Ensure it's an absolute path so monitor.py works anywhere
            selected_source = os.path.abspath(selected_source)
        elif choice == webcam_option:
            selected_source = 0
        else:
            print("Invalid choice. Exiting.")
            return
            
        print(f"\nStarting monitor with source: {selected_source}")
        print("Press 'q' in the video window to stop.")
        
        # Start the monitoring system
        start_system(video_sources=[selected_source])
        
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
