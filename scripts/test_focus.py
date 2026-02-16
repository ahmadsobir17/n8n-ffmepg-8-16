import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add script dir to path
sys.path.append('/opt/n8n-clipper/scripts')
from clipping_style1 import FaceDetector

def test_video(video_path):
    print(f"Testing video: {video_path}")
    detector = FaceDetector(min_confidence=0.3)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    talking_history = {'left': [], 'right': []}
    
    while frame_count < 300: # Test first 10 seconds (at 30fps)
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % 5 != 0: continue
        
        faces = detector.detect_faces(frame)
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        frame_info = []
        for i, f in enumerate(faces):
            side = 'left' if f['center_x'] < 0.5 else 'right'
            frame_info.append(f"F{i}({side}): x={f['center_x']:.2f}, y={f['center_y']:.2f}, open={f['openness']:.2f}")
            talking_history[side].append(f['openness'])
            
        if frame_info:
            print(f"[{ts:.2f}s] {' | '.join(frame_info)}")
            
    # Calculate stats
    for side in ['left', 'right']:
        if talking_history[side]:
            scores = talking_history[side]
            avg = sum(scores) / len(scores)
            variance = np.var(scores)
            print(f"\nStats for {side}:")
            print(f"  Avg Openness: {avg:.3f}")
            print(f"  Variance:     {variance:.6f}")
            print(f"  Max-Min:      {max(scores) - min(scores):.3f}")

    cap.release()
    detector.close()

if __name__ == "__main__":
    test_path = "/tmp/clipper/full_96574478-a10d-4a14-b3eb-607e169b0fec.mp4"
    if os.path.exists(test_path):
        test_video(test_path)
    else:
        print(f"File not found: {test_path}")
