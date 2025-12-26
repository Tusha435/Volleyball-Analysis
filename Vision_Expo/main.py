"""
Production volleyball tracking system with position-locked referee identification.

This is the main entry point for the volleyball tracking application. It initializes
the video capture, sets up the VolleyballTrackerPro system, processes each frame,
and outputs the annotated video with tracked players, referees, and ball.

The system uses a modular architecture combining YOLO detection, custom tracking
algorithms, camera motion compensation, and optional SAM2 segmentation for robust
multi-object tracking in volleyball game scenarios.
"""
import cv2
from config import CONFIG
from volleyball_tracker import VolleyballTrackerPro


def main():
    """
    Main execution function for the volleyball tracking application.

    This function orchestrates the complete tracking workflow including:
    1. Video loading and validation
    2. Tracker initialization with manual setup
    3. Frame-by-frame processing with real-time visualization
    4. Output video generation
    5. Error logging and cleanup

    The function handles all user interaction including the initial manual selection
    of net endpoints and referee positions, as well as runtime controls for quitting
    and capturing screenshots.

    Key Controls:
        'q': Quit processing and save output
        's': Save screenshot of current frame

    Returns:
        None. Outputs are saved to the configured output path.
    """
    print("="*70)
    print("PRODUCTION VOLLEYBALL TRACKER - POSITION-LOCKED REFEREES")
    print("="*70)

    cap = cv2.VideoCapture(CONFIG['video_path'])
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return

    ret, first = cap.read()
    if not ret:
        print("ERROR: Cannot read video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}FPS, {total} frames\n")

    tracker = VolleyballTrackerPro()
    if not tracker.setup(first):
        print("Setup cancelled")
        return

    writer = None
    if CONFIG['output_path']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(CONFIG['output_path'], fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = CONFIG['output_path'].replace('.mp4', '.avi')
            writer = cv2.VideoWriter(output, fourcc, fps, (width, height))
            if writer.isOpened():
                CONFIG['output_path'] = output
            else:
                writer = None

    print("\nProcessing... Press 'q' to quit\n")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % 30 == 0:
                pct = 100 * frame_id / total
                print(f"Frame {frame_id}/{total} ({pct:.1f}%)")

            tracker.process_frame(frame, frame_id)
            vis = tracker.visualize(frame)

            if writer:
                writer.write(vis)

            cv2.imshow("Volleyball Tracker - Position Locked", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f"screenshot_{frame_id}.jpg", vis)

    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\nSaved: {CONFIG['output_path']}")
        cv2.destroyAllWindows()
        tracker.save_error_report()

    print("\n"+"="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
