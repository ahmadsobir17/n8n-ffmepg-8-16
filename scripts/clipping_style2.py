#!/usr/bin/env python3
"""
Clipping Style 2 — CapCut-Style Face Tracking + Word-by-Word Subtitles

Inspired by yt-short-clipper, adapted for n8n-clipper stack.

Key differences from Style 1:
  1. CapCut-style word-by-word subtitle highlighting
  2. Frozen-frame hook intro (first frame + gold text on white box)
  3. MediaPipe Face Mesh lip activity tracking (active speaker detection)
  4. Closeup only mode (no split-screen)
  5. No speed ramp (1.0x natural audio)
  6. Gradient progress bar (white → yellow)
  7. Warmer color grading
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json
import random
import math

# Suppress logging & force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['EGL_LOG_LEVEL'] = 'fatal'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# Thread optimization for 8-core CPU
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['VECLIB_MAXIMUM_THREADS'] = '6'
os.environ['NUMEXPR_NUM_THREADS'] = '6'

try:
    import cv2
    import mediapipe as mp
    import whisper
    import torch
    import gc
    import numpy as np

    cv2.setNumThreads(6)
    torch.set_num_threads(6)
except ImportError as e:
    print(f"Error: {e}")
    print("Install: pip install opencv-python mediapipe openai-whisper torch numpy")
    sys.exit(1)

# Working directory & assets
WORK_DIR = '/tmp/clipper'
ASSETS_DIR = Path(__file__).parent / 'assets'
os.makedirs(WORK_DIR, exist_ok=True)

# SFX files
WHOOSH_FILES = [
    'Whooosh Swoosh - 9.wav',
    'Whoosh Swoosh - 21.wav',
    'Whoosh Swoosh - 6.wav',
    'Whoosh-1.wav',
    'Whoosh Swoosh-30.wav',
]
BG_MUSIC_FILE = 'diedlonely - in the bleak midwinter.mp3'

# Keyword maps (same as style1 for consistency)
SFX_MAP = {
    'BOOM': 'vine_boom.mp3',
    'WOW': 'ding.mp3',
    'BAGUS': 'ding.mp3',
    'KEREN': 'ding.mp3',
    'MANTAP': 'ding.mp3',
    'HEBAT': 'ding.mp3',
    'GILA': 'vine_boom.mp3',
    'BUSET': 'vine_boom.mp3',
}

EMOJI_MAP = {
    'UANG': '💰', 'DUIT': '💰', 'CUAN': '💰', 'PROFIT': '💰',
    'SAHAM': '📈', 'NAIK': '📈', 'TURUN': '📉', 'JATUH': '📉',
    'BULLISH': '🐂', 'BEARISH': '🐻', 'CRYPTO': '🪙', 'BITCOIN': '🪙',
    'BAHAYA': '⚠️', 'HATI-HATI': '⚠️', 'PERINGATAN': '⚠️',
    'API': '🔥', 'PANAS': '🔥', 'HOT': '🔥',
}

HIGHLIGHT_KEYWORDS = {
    'CUAN', 'PROFIT', 'SAHAM', 'BULLISH', 'BEARISH', 'CRYPTO',
    'BITCOIN', 'BAHAYA', 'GILA', 'BOOM', 'WOW', 'KEREN', 'MANTAP',
    'INVESTASI', 'PANI', 'RAJA', 'MSCI', 'DANA', 'MODAL',
    'BETOT', 'NAGA', 'BOCAH', 'SUHU', 'NARASI'
}

# MediaPipe Face Mesh lip landmark indices
# Upper lip outer: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
# Lower lip outer: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
UPPER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_IDX = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
# Simplified: use key points for aperture measurement
LIP_TOP = 13      # Upper lip center
LIP_BOTTOM = 14   # Lower lip center
LIP_LEFT = 61
LIP_RIGHT = 291


# ============================================
# FUZZY KEYWORD MATCHING
# ============================================
def fuzzy_keyword_match(word, map_dict, threshold=None):
    """Match a word against a keyword dictionary, handling punctuation and variations."""
    clean = word.strip().upper().rstrip('.,!?;:\'"')
    # Direct match
    if clean in map_dict:
        return clean, map_dict[clean]
    # Substring match (e.g., 'CUANNYA' contains 'CUAN')
    for key in map_dict:
        if key in clean and len(key) >= 3:
            return key, map_dict[key]
    return None, None


# ============================================
# FACE MESH DETECTOR (Lip Activity Tracking)
# ============================================
class FaceDetectorMesh:
    """MediaPipe Face Mesh wrapper with lip activity scoring.
    
    Instead of just detecting face bounding boxes, this uses the 468-point
    Face Mesh to measure lip aperture (jaw opening), enabling active speaker
    detection by identifying which face has the most lip movement.
    """
    
    def __init__(self, max_faces=2, min_confidence=0.4):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.3
        )
    
    def close(self):
        if hasattr(self, 'mesh') and self.mesh:
            self.mesh.close()
    
    def detect_faces(self, frame):
        """Detect faces and compute lip activity for each.
        
        Returns list of dicts with:
          - center_x, center_y: face center (0-1 normalized)
          - rel_w, rel_h: face bounding box size (0-1 normalized)
          - lip_aperture: how open the mouth is (0-1, higher = more open)
          - bbox: (x, y, w, h) in pixels
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        
        faces = []
        if not results.multi_face_landmarks:
            return faces
        
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Get bounding box from all landmarks
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            face_w = max_x - min_x
            face_h = max_y - min_y
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            # Compute lip aperture (vertical distance between upper and lower lip)
            lip_top = landmarks[LIP_TOP]
            lip_bottom = landmarks[LIP_BOTTOM]
            lip_left = landmarks[LIP_LEFT]
            lip_right = landmarks[LIP_RIGHT]
            
            # Vertical aperture normalized by lip width
            lip_width = math.sqrt((lip_right.x - lip_left.x)**2 + (lip_right.y - lip_left.y)**2)
            lip_height = math.sqrt((lip_bottom.x - lip_top.x)**2 + (lip_bottom.y - lip_top.y)**2)
            
            lip_aperture = lip_height / max(lip_width, 0.001)  # ratio, ~0.05 closed, ~0.5 open
            
            faces.append({
                'center_x': center_x,
                'center_y': center_y,
                'rel_w': face_w,
                'rel_h': face_h,
                'lip_aperture': lip_aperture,
                'bbox': (int(min_x * w), int(min_y * h), int(face_w * w), int(face_h * h))
            })
        
        # Sort by x position (left to right)
        faces.sort(key=lambda f: f['center_x'])
        return faces


# ============================================
# VIDEO UTILITIES
# ============================================
def get_video_info(video_path):
    """Get video metadata using ffprobe."""
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
           '-show_streams', '-show_format', video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        for s in info.get('streams', []):
            if s.get('codec_type') == 'video':
                return {
                    'width': int(s.get('width', 1920)),
                    'height': int(s.get('height', 1080)),
                    'fps': eval(s.get('r_frame_rate', '30/1')),
                    'duration': float(info.get('format', {}).get('duration', 0))
                }
    except Exception:
        pass
    return {'width': 1920, 'height': 1080, 'fps': 30, 'duration': 0}


def extract_segment_h264(video_path, start_time, duration):
    """Extract video segment with h264 encoding for analysis."""
    temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, dir=WORK_DIR)
    temp.close()
    cmd = [
        'ffmpeg', '-y', '-threads', '2',
        '-ss', str(start_time), '-t', str(duration),
        '-i', video_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-an', '-sn',
        '-vf', 'fps=5',  # 5fps for analysis (saves CPU)
        temp.name
    ]
    subprocess.run(cmd, capture_output=True)
    return temp.name


def smooth_coordinates(data, window_size=9):
    """Simple moving average smoothing for face coordinates."""
    if len(data) < window_size:
        return data
    smoothed = []
    half = window_size // 2
    for i in range(len(data)):
        start = max(0, i - half)
        end = min(len(data), i + half + 1)
        smoothed.append(sum(data[start:end]) / (end - start))
    return smoothed


# ============================================
# LIP ACTIVITY SPEAKER DETECTION
# ============================================
def compute_active_speaker_lip(frame_data, window_sec=1.5, fps=5):
    """Determine active speaker per frame using lip aperture variance.
    
    For each frame, looks at a sliding window of lip aperture values
    per face. The face with highest variance in lip aperture is likely
    the one talking (jaw moving up/down).
    
    Args:
        frame_data: list of frame dicts with 'all_faces' containing lip_aperture
        window_sec: sliding window duration in seconds
        fps: analysis frame rate
    
    Returns:
        Enriched frame_data with 'active_speaker_x' and 'active_speaker_w' per frame
    """
    window = max(3, int(window_sec * fps))
    
    for i, frame in enumerate(frame_data):
        faces = frame.get('all_faces', [])
        if not faces:
            frame['active_speaker_x'] = 0.5
            frame['active_speaker_w'] = 0.2
            continue
        
        if len(faces) == 1:
            frame['active_speaker_x'] = faces[0]['center_x']
            frame['active_speaker_w'] = faces[0].get('rel_w', 0.15)
            continue
        
        # Multiple faces: compute lip aperture variance per face in window
        start_idx = max(0, i - window // 2)
        end_idx = min(len(frame_data), i + window // 2 + 1)
        
        best_face_idx = 0
        best_variance = -1
        
        for fi in range(len(faces)):
            apertures = []
            for j in range(start_idx, end_idx):
                other_faces = frame_data[j].get('all_faces', [])
                if fi < len(other_faces):
                    apertures.append(other_faces[fi].get('lip_aperture', 0))
            
            if len(apertures) >= 2:
                mean = sum(apertures) / len(apertures)
                variance = sum((a - mean)**2 for a in apertures) / len(apertures)
            else:
                variance = 0
            
            if variance > best_variance:
                best_variance = variance
                best_face_idx = fi
        
        active = faces[min(best_face_idx, len(faces) - 1)]
        frame['active_speaker_x'] = active['center_x']
        frame['active_speaker_w'] = active.get('rel_w', 0.15)
    
    return frame_data


# ============================================
# AI VIDEO ANALYSIS (Face Mesh)
# ============================================
def analyze_video_facemesh(video_path, start_time, duration):
    """Analyze video using MediaPipe Face Mesh for lip-based speaker tracking.
    
    Always produces closeup-mode segments (no split-screen).
    """
    print(f"  [1/4] Analyzing video (Face Mesh mode)...", flush=True)
    
    # Get video info
    info = get_video_info(video_path)
    w, h = info['width'], info['height']
    print(f"    📐 Video: {w}x{h}", flush=True)
    
    # Extract segment at 5fps for analysis
    temp_seg = extract_segment_h264(video_path, start_time, duration)
    
    cap = cv2.VideoCapture(temp_seg)
    if not cap.isOpened():
        print("    ✗ Cannot open video segment", flush=True)
        os.unlink(temp_seg)
        return None
    
    analysis_fps = cap.get(cv2.CAP_PROP_FPS) or 5
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"    📊 Analyzing {total_frames} frames at {analysis_fps:.1f}fps", flush=True)
    
    detector = FaceDetectorMesh(max_faces=2, min_confidence=0.4)
    frame_data = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        ts = frame_idx / analysis_fps
        faces = detector.detect_faces(frame)
        
        primary = faces[0] if faces else {'center_x': 0.5, 'center_y': 0.5, 'rel_w': 0, 'rel_h': 0, 'lip_aperture': 0}
        
        frame_data.append({
            'timestamp': ts,
            'primary_x': primary['center_x'],
            'face_area': primary['rel_w'] * primary['rel_h'],
            'all_faces': faces,
            'mode': 'closeup'  # Always closeup in style2
        })
        
        frame_idx += 1
        del frame
    
    cap.release()
    os.unlink(temp_seg)
    detector.close()
    
    if not frame_data:
        print("    ✗ No frames analyzed", flush=True)
        return None
    
    # Smooth X coordinates
    x_coords = [f['primary_x'] for f in frame_data]
    smoothed_x = smooth_coordinates(x_coords, window_size=11)
    for i, f in enumerate(frame_data):
        f['smoothed_x'] = smoothed_x[i]
    
    # Compute active speaker using lip activity
    frame_data = compute_active_speaker_lip(frame_data, window_sec=1.5, fps=analysis_fps)
    print(f"    🎯 Active speaker timeline computed ({len(frame_data)} frames)", flush=True)
    
    # Build single closeup segment covering full duration
    segments = [{'start': 0, 'end': duration, 'mode': 'closeup'}]
    
    # Enrich segment with average speaker position
    avg_x = sum(f.get('active_speaker_x', f['smoothed_x']) for f in frame_data) / len(frame_data)
    segments[0]['avg_x'] = avg_x
    segments[0]['speaker1_x'] = avg_x
    segments[0]['speaker2_x'] = avg_x
    
    print(f"  [1/4] Analysis complete — Closeup mode, avg speaker @ {avg_x:.2f}", flush=True)
    
    return {
        'video_width': w,
        'video_height': h,
        'segments': segments,
        'adjusted_start_time': start_time,
        '_frame_data': frame_data
    }


# ============================================
# WHISPER TRANSCRIPTION
# ============================================
def transcribe_audio(video_path, start_time, duration):
    """Transcribe audio using local Whisper model."""
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=WORK_DIR)
    temp.close()
    try:
        cmd = [
            'ffmpeg', '-y', '-threads', '1',
            '-ss', str(start_time), '-t', str(duration),
            '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            temp.name
        ]
        subprocess.run(cmd, capture_output=True)
        
        print(f"  [2/4] Loading Whisper model (small)...", flush=True)
        model = whisper.load_model('small')
        print(f"  [2/4] Transcribing audio...", flush=True)
        result = model.transcribe(temp.name, word_timestamps=True, language='id', fp16=False)
        words = []
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                words.append({'word': word['word'].strip(), 'start': word['start'], 'end': word['end']})
        
        print(f"  [2/4] Transcribed {len(words)} words", flush=True)
        
        del model, result
        gc.collect()
        return words
    finally:
        if os.path.exists(temp.name):
            os.unlink(temp.name)


# ============================================
# SFX EVENT DETECTION (Enhanced)
# ============================================
def get_enhanced_sfx_events(words, has_hook_title=False):
    """Generate context-aware SFX events.
    
    1. Pop on hook title appearance
    2. Ding on highlight words (rate-limited: max 1 per 3.5s)
    3. Keyword-triggered SFX
    """
    events = []
    
    # 1. Hook title pop
    if has_hook_title:
        pop_path = str(ASSETS_DIR / 'pop.mp3')
        if os.path.exists(pop_path):
            events.append({'file': pop_path, 'time': 0.2, 'keyword': 'HOOK_POP', 'volume': 0.25})
    
    # 2. Ding on highlight words (rate-limited)
    ding_path = str(ASSETS_DIR / 'ding.mp3')
    last_ding_t = -999
    if os.path.exists(ding_path):
        for w in words:
            clean = w['word'].strip().upper().rstrip('.,!?;:')
            if clean in HIGHLIGHT_KEYWORDS and w['start'] - last_ding_t >= 3.5:
                events.append({'file': ding_path, 'time': w['start'], 'keyword': f'HIGHLIGHT:{clean}', 'volume': 0.15})
                last_ding_t = w['start']
    
    # 3. Keyword-triggered SFX
    existing_times = {e['time'] for e in events}
    for w in words:
        k, v = fuzzy_keyword_match(w['word'], SFX_MAP)
        if k and w['start'] not in existing_times:
            sfx_path = str(ASSETS_DIR / v)
            if os.path.exists(sfx_path):
                events.append({'file': sfx_path, 'time': w['start'], 'keyword': k, 'volume': 0.20})
                existing_times.add(w['start'])
    
    events.sort(key=lambda e: e['time'])
    return events


# ============================================
# FFMPEG FILTER: HOOK INTRO SCENE
# ============================================
def build_hook_filter(hook_title, hook_duration=3.0):
    """Build FFmpeg filter for frozen-frame hook intro.
    
    Creates a scene by:
    1. Looping the first frame for hook_duration seconds
    2. Overlaying multi-line gold text on white semi-transparent boxes
    
    Returns: (filter_string, hook_duration)
    The filter takes [cropped] as input and outputs [hooked].
    """
    if not hook_title:
        return "", 0
    
    # Format text: uppercase, split into lines (max 3 words per line)
    hook_upper = hook_title.upper()
    words = hook_upper.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(current_line) >= 3:
            lines.append(' '.join(current_line))
            current_line = []
    if current_line:
        lines.append(' '.join(current_line))
    
    # Build drawtext filters for each line
    drawtext_parts = []
    line_height = 85
    font_size = 56
    total_text_height = len(lines) * line_height
    start_y = 640 - (total_text_height // 2)  # Upper third of 1920
    
    for i, line in enumerate(lines):
        # Escape special FFmpeg characters
        escaped = line.replace("'", "'\\''").replace(":", "\\:").replace("\\", "\\\\")
        y_pos = start_y + (i * line_height)
        
        drawtext_parts.append(
            f"drawtext=text='{escaped}':"
            f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"fontsize={font_size}:"
            f"fontcolor=#FFD700:"  # Golden yellow
            f"box=1:"
            f"boxcolor=white@0.90:"
            f"boxborderw=14:"
            f"x=(w-text_w)/2:"
            f"y={y_pos}"
        )
    
    drawtext_chain = ",".join(drawtext_parts)
    
    # Filter: split → freeze first frame as hook → apply text → concat with main
    # Fade in text (alpha 0→1 in first 0.3s), fade out at end
    hook_filter = (
        f"[cropped]split=2[hook_src][main_src];"
        f"[hook_src]trim=0:0.04,loop=loop={int(hook_duration * 30)}:size=1:start=0,"
        f"setpts=N/30/TB,{drawtext_chain},"
        f"trim=0:{hook_duration},setpts=PTS-STARTPTS[hook_v];"
        f"[main_src]setpts=PTS-STARTPTS[main_v];"
        f"[hook_v][main_v]concat=n=2:v=1:a=0[hooked]"
    )
    
    return hook_filter, hook_duration


# ============================================
# FFMPEG FILTER: CROP + TRACKING
# ============================================
def generate_crop_filter(analysis, duration, words, tracking=True, hook_title=None):
    """Generate FFmpeg filter_complex for closeup crop with lip-based tracking.
    
    Always closeup mode — no split-screen logic.
    """
    vid_w, vid_h = analysis['video_width'], analysis['video_height']
    segments = analysis['segments']
    if not segments:
        return "nullsink", "[0:a]anull"
    
    out_w, out_h = 1080, 1920
    aspect_9_16 = 9 / 16
    target_fps = 30
    
    # No speed factor in style2
    speed_factor = 1.0
    
    # Calculate crop dimensions
    target_h = vid_h
    target_w = int(vid_h * aspect_9_16)
    cw = (target_w // 2) * 2  # ensure even
    ch = (target_h // 2) * 2
    
    # Build face tracking x expression
    seg = segments[0]  # Single closeup segment
    seg_frames = analysis.get('_frame_data', [])
    
    if not tracking or not seg_frames:
        # Static center on average face position  
        avg_x = seg.get('avg_x', 0.5)
        xb = max(0, min(int(avg_x * vid_w) - target_w // 2, vid_w - target_w))
        x_expr = str(xb)
    else:
        # Dynamic tracking using active_speaker_x (lip-detected)
        tracking_points = []
        last_t = -1
        for f in seg_frames:
            if f['timestamp'] >= last_t + 0.4:  # sample every 0.4s
                speaker_x = f.get('active_speaker_x', f['smoothed_x'])
                face_w = f.get('active_speaker_w', 0.1)
                
                # Face-aware clamping
                face_half_w_rel = (face_w * 1.15) / 2
                min_x = face_half_w_rel + (target_w / 2) / vid_w
                max_x = 1.0 - face_half_w_rel - (target_w / 2) / vid_w
                min_x = max(min_x, (target_w / 2) / vid_w)
                max_x = min(max_x, 1.0 - (target_w / 2) / vid_w)
                
                if min_x < max_x:
                    clamped_x = max(min_x, min(max_x, speaker_x))
                else:
                    clamped_x = 0.5
                
                tracking_points.append((f['timestamp'], clamped_x))
                last_t = f['timestamp']
        
        if len(tracking_points) < 2:
            xb = max(0, min(int(seg.get('avg_x', 0.5) * vid_w) - target_w // 2, vid_w - target_w))
            x_expr = str(xb)
        else:
            # Build lerp expression for smooth panning
            parts = []
            first_x = max(0, min(int(tracking_points[0][1] * vid_w) - target_w // 2, vid_w - target_w))
            for j in range(len(tracking_points) - 1):
                t1, x1 = tracking_points[j]
                t2, x2 = tracking_points[j + 1]
                p1 = max(0, min(int(x1 * vid_w) - target_w // 2, vid_w - target_w))
                p2 = max(0, min(int(x2 * vid_w) - target_w // 2, vid_w - target_w))
                rel_t1 = round(t1, 3)
                rel_t2 = round(t2, 3)
                dur_t = round(t2 - t1, 3)
                lerp = str(p1) if p1 == p2 else f"({p1}+({p2}-{p1})*(t-{rel_t1})/{dur_t})"
                if j == len(tracking_points) - 2:
                    parts.append(f"({lerp}*between(t,{rel_t1},{rel_t2}))")
                else:
                    parts.append(f"({lerp}*gte(t,{rel_t1})*lt(t,{rel_t2}))")
            first_rel_t = round(tracking_points[0][0], 3)
            x_expr = f"trunc({first_x}*lt(t,{first_rel_t}) + {' + '.join(parts)})"
    
    final_x = f"min(max(0,{x_expr}),iw-{cw})"
    
    # Main video filter: trim → crop → scale → color grade → progress bar
    f_str = (
        f"[0:v]fps={target_fps},"
        f"crop=w={cw}:h={ch}:x='{final_x}':y='(ih-oh)/2',"
        f"scale={out_w}:{out_h}:flags=lanczos,"
        f"setsar=1,format=yuv420p[cropped]"
    )
    
    # Hook intro (frozen frame + text)
    hook_filter, hook_dur = build_hook_filter(hook_title)
    if hook_filter:
        f_str += f";{hook_filter}"
        video_label = "[hooked]"
        total_duration = duration + hook_dur
    else:
        video_label = "[cropped]"
        total_duration = duration
    
    # Color grading: warmer tones (slightly more saturation, warm brightness)
    f_str += (
        f";{video_label}eq=contrast=1.12:saturation=1.25:brightness=-0.03[graded_base]"
    )
    
    # Gradient progress bar (white → yellow)
    f_str += (
        f";color=c=white:s=1080x6[pbar_w]"
        f";color=c=yellow@0.9:s=1080x6[pbar_y]"
        f";[pbar_w][pbar_y]blend=all_mode=normal:all_opacity=0.5[pbar]"
        f";[graded_base][pbar]overlay=x='-w+(w*t/{total_duration:.2f})':y=H-6:shortest=1[graded]"
    )
    
    # Audio filter (no speed change)
    a_str = "[0:a]aresample=async=1,atrim=start=0:end=" + str(duration) + ",asetpts=PTS-STARTPTS"
    
    # If hook exists, add silence pad at beginning of audio
    if hook_dur > 0:
        a_str += f",adelay={int(hook_dur * 1000)}|{int(hook_dur * 1000)}"
    
    a_str += "[main_audio]"
    
    return f_str, a_str, hook_dur


# ============================================
# ASS SUBTITLE: CAPCUT STYLE
# ============================================
def format_ass_time(s):
    """Format seconds to ASS time format H:MM:SS.cc"""
    s = max(0, s)
    return f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}.{int((s % 1) * 100):02d}"


def generate_capcut_subtitle(words, title=None, hook_duration=0.0):
    """Generate CapCut-style ASS subtitle with word-by-word highlighting.
    
    For each 4-word chunk, generates separate dialogue events for each word,
    showing the full chunk but highlighting only the currently-spoken word in yellow.
    This creates the signature 'karaoke' effect seen in CapCut/TikTok edits.
    """
    if not words:
        return ""
    
    # ASS Header — Bold text, thick outline, positioned at lower quarter
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Default: white bold, thick black outline, bottom-center
        "Style: Default,DejaVu Sans,72,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,2,0,1,5,3,2,50,50,200,1\n"
        # Hook: large golden, thick border, positioned upper-third
        "Style: Hook,DejaVu Sans,72,&H0000D7FF,&H00FFFFFF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,4,0,1,6,0,5,60,60,100,1\n"
        # Watermark: small white, semi-transparent, top-right
        "Style: Watermark,DejaVu Sans,30,&H50FFFFFF,&H50FFFFFF,&H60000000,&H00000000,"
        "-1,0,0,0,100,100,1,0,1,2,1,9,0,30,25,1\n"
        # Source: small white, bottom-left
        "Style: Source,DejaVu Sans,24,&H70FFFFFF,&H70FFFFFF,&H80000000,&H00000000,"
        "0,0,0,0,100,100,1,0,1,1,0,1,25,0,25,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    events = ""
    
    # Hook title with pop animation (during hook scene)
    if title and hook_duration > 0:
        pop = (
            r"{\pos(540,350)"
            r"\fscx100\fscy100"
            r"\t(0,200,\fscx115\fscy115)"
            r"\t(200,400,\fscx100\fscy100)"
            r"\be3"
            r"\bord8"
            r"\3c&H000000&}"
        )
        events += f"Dialogue: 1,0:00:00.00,{format_ass_time(hook_duration)},Hook,,0,0,0,,{pop}{title.upper()}\n"
    
    # Offset all word timestamps by hook_duration
    time_offset = hook_duration
    
    # Group words into 4-word chunks
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        
        # For each word in the chunk, create a subtitle event highlighting that word
        for j, current_word in enumerate(chunk):
            word_start = current_word['start'] + time_offset
            word_end = current_word['end'] + time_offset
            
            # Ensure minimum duration of 0.1s for visibility
            if word_end - word_start < 0.1:
                word_end = word_start + 0.1
            
            # Build text with current word highlighted in yellow
            text_parts = []
            for k, w in enumerate(chunk):
                word_text = w['word'].strip().upper()
                
                # Check for emoji
                _, emoji = fuzzy_keyword_match(word_text, EMOJI_MAP)
                emoji_str = f" {emoji}" if emoji else ""
                
                if k == j:
                    # Highlight current word (yellow: &H00FFFF in BGR format)
                    text_parts.append(f"{{\\c&H00FFFF&}}{word_text}{emoji_str}{{\\c&HFFFFFF&}}")
                else:
                    text_parts.append(f"{word_text}{emoji_str}")
            
            text = ' '.join(text_parts)
            
            # Pop-in animation for each subtitle event
            pop_tag = r"\fscx110\fscy110\t(0,80,\fscx100\fscy100)"
            
            events += (
                f"Dialogue: 0,{format_ass_time(word_start)},{format_ass_time(word_end)},"
                f"Default,,0,0,0,,{{{pop_tag}\\pos(540,1550)}}{text}\n"
            )
    
    return header + events


def generate_watermark_events(duration, watermark='@visi.bisinis', channel_name=None, hook_duration=0.0):
    """Generate ASS watermark events."""
    events = ""
    total_dur = duration + hook_duration
    end_time = format_ass_time(total_dur)
    
    if watermark:
        pulse = (
            r"\fscx100\fscy100"
            r"\t(0,2000,\fscx102\fscy102)"
            r"\t(2000,4000,\fscx100\fscy100)"
        )
        events += (
            f"Dialogue: 2,0:00:00.00,{end_time},Watermark,,0,0,0,,"
            f"{{{pulse}}}{watermark}\n"
        )
    
    if channel_name:
        source_text = f"src: {channel_name}"
        events += (
            f"Dialogue: 2,0:00:00.00,{end_time},Source,,0,0,0,,"
            f"{source_text}\n"
        )
    
    return events


# ============================================
# MAIN VIDEO PROCESSING
# ============================================
def process_video(input_file, output_file, start_time, duration, ass_file=None,
                  hook_title=None, ai_mode='auto', tracking=True,
                  watermark='@visi.bisinis', channel_name=None):
    """Main processing function — Style 2: CapCut + Face Mesh.
    
    Same interface as clipping_style1.py for n8n compatibility.
    """
    print(f"\n{'='*50}", flush=True)
    print(f"  VIDEO PROCESSING START (Style 2 — CapCut)", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Input:     {input_file}", flush=True)
    print(f"  Output:    {output_file}", flush=True)
    print(f"  Start:     {start_time}s, Duration: {duration}s", flush=True)
    print(f"  Tracking:  {tracking}", flush=True)
    print(f"  Hook:      {hook_title or '(none)'}", flush=True)
    print(f"  Watermark: {watermark or '(none)'}", flush=True)
    print(f"  Source:    {channel_name or '(none)'}", flush=True)

    # Validate input
    if not os.path.exists(input_file):
        print(f"  ✗ Error: Input file not found: {input_file}", flush=True)
        sys.exit(1)

    # ── Step 1: AI Video Analysis (Face Mesh) ──
    analysis = analyze_video_facemesh(input_file, start_time, duration)
    if not analysis:
        print("  ✗ Error: Video analysis failed", flush=True)
        sys.exit(1)
    
    actual_start = analysis['adjusted_start_time']

    # ── Step 2: Whisper Transcription ──
    words = transcribe_audio(input_file, actual_start, duration)
    gc.collect()

    # ── Step 3: Effect Preparation ──
    print(f"  [3/4] Preparing effects...", flush=True)
    
    # SFX events
    sfx_events = get_enhanced_sfx_events(words, has_hook_title=bool(hook_title))
    print(f"    SFX triggers: {len(sfx_events)}", flush=True)
    for sfx in sfx_events:
        print(f"      → {sfx['keyword']} at {sfx['time']:.1f}s vol={sfx.get('volume',0.2):.0%} ({os.path.basename(sfx['file'])})", flush=True)
    
    # Background music
    bg_music_path = str(ASSETS_DIR / BG_MUSIC_FILE)
    has_bg_music = os.path.exists(bg_music_path)
    if has_bg_music:
        print(f"    🎵 Background music: {BG_MUSIC_FILE}", flush=True)
    
    # Generate filter_complex
    f_complex, a_filter, hook_dur = generate_crop_filter(
        analysis, duration, words, tracking=tracking, hook_title=hook_title
    )
    v_stream = "[graded]"
    total_duration = duration + hook_dur
    
    # ASS subtitle overlay (CapCut style)
    if ass_file:
        ass_content = generate_capcut_subtitle(words, title=hook_title, hook_duration=hook_dur)
        ass_content += generate_watermark_events(
            duration, watermark=watermark, channel_name=channel_name, hook_duration=hook_dur
        )
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        print(f"    ASS subtitle: {ass_file} ({len(ass_content)} bytes)", flush=True)
        
        ass_escaped = ass_file.replace('\\', '/').replace("'", "\\'").replace(":", "\\:")
        f_complex += f";[graded]ass='{ass_escaped}'[outv]"
        v_stream = "[outv]"
    
    # ── Audio mixing ──
    # Full filter = video filter + audio filter
    full_filter = f_complex + ";" + a_filter
    a_stream = "[main_audio]"
    
    sfx_filter, sfx_inputs = "", []
    next_input_idx = 1
    
    if sfx_events:
        for i, sfx in enumerate(sfx_events):
            # Offset SFX times by hook duration
            rel_t = max(0, (sfx['time'] + hook_dur) * 1000)
            vol = sfx.get('volume', 0.20)
            sfx_inputs.extend(['-i', sfx['file']])
            sfx_filter += f"[{next_input_idx}:a]adelay={rel_t:.0f}|{rel_t:.0f},volume={vol}[sfx{i}];"
            next_input_idx += 1
        
        sfx_refs = ''.join([f'[sfx{i}]' for i in range(len(sfx_events))])
        sfx_filter += f"{a_stream}{sfx_refs}amix=inputs={len(sfx_events) + 1}:normalize=0[sfx_mixed];"
        a_stream = "[sfx_mixed]"
    
    # Background music
    bg_inputs = []
    if has_bg_music:
        bg_inputs.extend(['-stream_loop', '-1', '-i', bg_music_path])
        bg_idx = next_input_idx
        fade_out_start = max(0, total_duration - 2.0)
        sfx_filter += (
            f"[{bg_idx}:a]atrim=0:{total_duration:.2f},"
            f"afade=t=in:d=1,afade=t=out:st={fade_out_start:.2f}:d=2,"
            f"volume=0.08[bgm];"
            f"{a_stream}[bgm]amix=inputs=2:normalize=0[final_audio];"
        )
        a_stream = "[final_audio]"
        if sfx_filter.endswith(';'):
            sfx_filter = sfx_filter[:-1]
    
    full_filter += (";" + sfx_filter if sfx_filter else "")
    
    # Debug output
    print(f"  --- Filter Complex ({len(full_filter)} chars, {full_filter.count(';')+1} stages) ---", flush=True)
    for idx, part in enumerate(full_filter.split(";")):
        truncated = part[:150] + ('...' if len(part) > 150 else '')
        print(f"    [{idx:2d}] {truncated}", flush=True)
    print(f"  --- End Filter Debug ---", flush=True)
    
    # ── Step 4: FFmpeg Render ──
    print(f"  [4/4] Rendering final video...", flush=True)
    
    try:
        cmd = [
            'ffmpeg', '-y', '-threads', '6',
            '-ss', str(actual_start), '-t', str(duration),
            '-i', input_file
        ] + sfx_inputs + bg_inputs + [
            '-filter_complex', full_filter,
            '-map', v_stream,
            '-map', a_stream,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
            '-max_muxing_queue_size', '4096',
            '-movflags', '+faststart',
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        
        if result.returncode != 0:
            print(f"  ✗ FFmpeg FAILED", flush=True)
            stderr_lines = result.stderr.strip().split('\n')
            error_lines = [l for l in stderr_lines if any(k in l.lower() for k in ['error', 'invalid', 'no such', 'failed', 'cannot'])]
            if not error_lines:
                error_lines = stderr_lines[-15:]
            print(f"  Error details:", flush=True)
            for line in error_lines:
                print(f"    {line.strip()}", flush=True)
            sys.exit(1)
        else:
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024 * 1024)
                print(f"  ✓ Success: {output_file} ({size_mb:.2f} MB)", flush=True)
            else:
                print(f"  ✗ Error: Output file not created", flush=True)
                sys.exit(1)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================
# CLI ENTRY POINT
# ============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Style 2 — CapCut Face Mesh Clipping')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--start', type=float, default=0)
    parser.add_argument('-d', '--duration', type=float, default=30)
    parser.add_argument('--ass-file')
    parser.add_argument('-t', '--title')
    parser.add_argument('--ai-mode', choices=['auto', 'prefer_closeup', 'prefer_split'], default='auto')
    parser.add_argument('-title', dest='title_alias')
    parser.add_argument('--tracking', action='store_true', default=True)
    parser.add_argument('--no-tracking', action='store_false', dest='tracking')
    parser.add_argument('--watermark', default='@visi.bisinis', help='Watermark text (top-right)')
    parser.add_argument('--channel-name', default=None, help='Source channel name (bottom-left)')
    
    args = parser.parse_args()
    
    if args.title_alias and not args.title:
        args.title = args.title_alias
    
    process_video(
        args.input,
        args.output,
        args.start,
        args.duration,
        args.ass_file,
        args.title,
        args.ai_mode,
        tracking=args.tracking,
        watermark=args.watermark if args.watermark else None,
        channel_name=args.channel_name
    )
    
    print(f"\n{'='*50}", flush=True)
    print(f"  PROCESSING COMPLETE! (Style 2)", flush=True)
    print(f"{'='*50}", flush=True)
