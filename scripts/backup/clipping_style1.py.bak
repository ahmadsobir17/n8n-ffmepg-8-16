#!/usr/bin/env python3
"""
Face Detection Auto-Crop for Podcast Videos
AI Dynamic Closeup/Split Detection + Premium Effects

Effects included:
  1. AI Face Detection & Tracking (MediaPipe)
  2. Dynamic Closeup/Split Mode switching
  3. Zoom Punch effect (periodic zoom burst)
  4. Sound Effects (keyword-triggered SFX)
  5. ASS Subtitles with pop animation
  6. Emoji overlays (via ASS)
  7. Keyword highlighting (yellow text)
  8. Hook title animation
  9. Vignette dark corners
  10. Color grading (contrast/saturation)
  11. Progress bar (yellow)
  12. Speed ramp (1.05x)
  13. Dissolve crossfades between segments
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json
import random

# Suppress logging dan force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['EGL_LOG_LEVEL'] = 'fatal'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# OPTIMIZED: Utilizing 6 threads for 8-core CPU (leaving overhead for system)
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

# Working directory
WORK_DIR = '/tmp/clipper'
ASSETS_DIR = Path(__file__).parent / 'assets'
os.makedirs(WORK_DIR, exist_ok=True)

def fuzzy_keyword_match(word, map_dict, threshold=None):
    """
    Match a word against a dictionary of keywords using simple fuzzy logic.
    Handles punctuation and variations like 'BULLISH!!!' or 'CUANNYA'.
    """
    if not word or not map_dict:
        return None, None
    
    clean_w = word.upper().strip('.,!?;:"\'-')
    if not clean_w:
        return None, None
        
    # 1. Direct match
    if clean_w in map_dict:
        return clean_w, map_dict[clean_w]
        
    # 2. Starts with / Ends with (handles "CUANNYA" or "DIBETOT")
    for k, v in map_dict.items():
        if clean_w.startswith(k) or clean_w.endswith(k):
            # Minimum length to avoid false positives (e.g., 'A' in 'API')
            if len(k) >= 3:
                return k, v
                
    return None, None

# ============================================
# EFFECT CONFIG: Keyword → SFX / Emoji Maps
# ============================================
SFX_MAP = {
    'BOOM': 'vine_boom.mp3',
    'WOW': 'ding.mp3',
    'BAGUS': 'ding.mp3',
    'KEREN': 'ding.mp3',
    'TIPS': 'pop.mp3',
    'RAHASIA': 'pop.mp3',
    'HEY': 'whoosh.mp3',
    'TRANSISI': 'whoosh.mp3',
    'CUAN': 'ding.mp3',
    'BETOT': 'whoosh.mp3',
    'HAJAR': 'vine_boom.mp3',
}

# NOTE: libass cannot render color emoji (Noto Color Emoji).
# Use ASCII-safe text symbols that DejaVu Sans can render.
EMOJI_MAP = {
    'DUIT': '[$$]',
    'CASH': '[$$]',
    'CUAN': '[$$]',
    'HATI': '<3',
    'SETUJU': '[OK]',
    'SALAH': '[X]',
    'API': '*',
    'NAGA': '>>',
    'SAHAM': '[+]',
    'BETOT': '>>',
    'BEARISH': '[-]',
    'BULLISH': '[+]',
}

HIGHLIGHT_KEYWORDS = {
    'CUAN', 'SAHAM', 'MARKET', 'PROFIT', 'LEVEL', 'OWNER',
    'BANDAR', 'RETAIL', 'BULLISH', 'BEARISH', 'TRADING',
    'INVESTASI', 'PANI', 'RAJA', 'MSCI', 'DANA', 'MODAL',
    'BETOT', 'NAGA', 'BOCAH', 'SUHU', 'NARASI'
}

# ============================================
# FACE DETECTION
# ============================================
class FaceDetector:
    def __init__(self, min_confidence=0.3):
        import mediapipe.python.solutions.face_detection as mp_face_detection
        self.mp_face_detection = mp_face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=min_confidence
        )
    
    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()
            del self.detector
        gc.collect()
    
    def detect_faces(self, frame):
        if frame is None:
            return []
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.detector.process(rgb_frame)
        except:
            return []
        
        faces = []
        if results.detections:
            for d in results.detections:
                bbox = d.location_data.relative_bounding_box
                faces.append({
                    'center_x': bbox.xmin + bbox.width / 2,
                    'center_y': bbox.ymin + bbox.height / 2,
                    'rel_w': bbox.width,
                    'rel_h': bbox.height
                })
        
        return faces

# ============================================
# VIDEO UTILITIES
# ============================================
def get_video_info(video_path):
    """Get video metadata"""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-of', 'json', video_path
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    try:
        data = json.loads(res.stdout)
        s = data['streams'][0]
        return int(s['width']), int(s['height']), float(s.get('duration', 0))
    except:
        return None, None, None

def extract_segment_h264(video_path, start_time, duration):
    """Extract video segment dengan h264 encoding for analysis"""
    temp = tempfile.NamedTemporaryFile(suffix='_seg.mp4', delete=False, dir=WORK_DIR)
    temp.close()
    
    cmd = [
        'ffmpeg', '-y', '-threads', '1',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '28',
        '-an',
        temp.name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    
    if result.returncode == 0:
        return temp.name
    
    print(f"  Error extracting segment: {result.stderr}", flush=True)
    return None

def smooth_coordinates(data, window_size=9):
    """Simple moving average smoothing for face coordinates"""
    if not data:
        return []
    if len(data) < window_size:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        avg = sum(data[start:end]) / (end - start)
        smoothed.append(avg)
    return smoothed

# ============================================
# AI VIDEO ANALYSIS
# ============================================
def analyze_video_dynamic(video_path, start_time, duration, ai_mode='auto'):
    """AI analysis for face detection and segmenting"""
    print(f"  [1/4] Analyzing video (mode: {ai_mode})...", flush=True)
    
    detector = FaceDetector(min_confidence=0.3)
    w, h, total_dur = get_video_info(video_path)
    if w is None: return None
    
    is_ultrawide = (w / h) > 2.0
    temp_seg = extract_segment_h264(video_path, start_time, duration)
    if not temp_seg: return None
    
    cap = cv2.VideoCapture(temp_seg)
    frame_data = []
    frame_count = 0
    dual_count = 0
    closeup_count = 0
    total_frames = 0
    
    CLOSEUP_FACE_AREA_THRESHOLD = 0.08
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % 3 != 0: continue
        
        total_frames += 1
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        faces = detector.detect_faces(frame)
        
        if len(faces) >= 2: dual_count += 1
        
        primary = faces[0] if faces else {'center_x': 0.5, 'center_y': 0.5, 'rel_w': 0, 'rel_h': 0}
        face_area = primary['rel_w'] * primary['rel_h']
        is_closeup_shot = face_area >= CLOSEUP_FACE_AREA_THRESHOLD
        if is_closeup_shot: closeup_count += 1
        
        # Mode decision
        if ai_mode == 'prefer_split':
            mode = 'wide' if (len(faces) >= 2 and not is_closeup_shot) else 'closeup'
        elif ai_mode == 'prefer_closeup':
            mode = 'wide' if (len(faces) >= 2 and dual_count > total_frames * 0.3 and not is_closeup_shot) else 'closeup'
        else:  # auto
            if is_closeup_shot: mode = 'closeup'
            elif len(faces) >= 2: mode = 'wide'
            elif is_ultrawide: mode = 'wide'
            else: mode = 'closeup'
            
        frame_data.append({
            'timestamp': ts,
            'primary_x': primary['center_x'],
            'face_area': face_area,
            'mode': mode
        })
        del frame

    x_coords = [f['primary_x'] for f in frame_data]
    smoothed_x = smooth_coordinates(x_coords, window_size=11)
    for i, f in enumerate(frame_data): f['smoothed_x'] = smoothed_x[i]

    cap.release()
    os.unlink(temp_seg)
    detector.close()
    
    # Build segments with minimum duration
    segments = []
    if frame_data:
        curr_mode, start_t = frame_data[0]['mode'], 0
        min_dur = 2.0
        for i in range(1, len(frame_data)):
            seg_dur = frame_data[i]['timestamp'] - start_t
            if frame_data[i]['mode'] != curr_mode and seg_dur >= min_dur:
                segments.append({'start': start_t, 'end': frame_data[i]['timestamp'], 'mode': curr_mode})
                start_t, curr_mode = frame_data[i]['timestamp'], frame_data[i]['mode']
        segments.append({'start': start_t, 'end': duration, 'mode': curr_mode})
    
    # ── FORCED MODE ALTERNATION ──
    # If >80% of the video is one mode (common in podcasts with 2 people),
    # force alternating segments for visual variety.
    total_dur = sum(s['end'] - s['start'] for s in segments)
    wide_dur = sum(s['end'] - s['start'] for s in segments if s['mode'] == 'wide')
    
    if total_dur > 15 and (wide_dur / total_dur > 0.80 or wide_dur / total_dur < 0.20):
        dominant = 'wide' if wide_dur / total_dur > 0.5 else 'closeup'
        alt = 'closeup' if dominant == 'wide' else 'wide'
        print(f"    ⚡ Forcing mode alternation ({dominant} dominant: {wide_dur/total_dur:.0%})", flush=True)
        
        # Create alternating segments: dominant 8-12s, alt 4-6s
        new_segments = []
        t = 0
        use_dominant = True
        while t < duration:
            if use_dominant:
                seg_len = min(random.uniform(8, 12), duration - t)
                new_segments.append({'start': t, 'end': t + seg_len, 'mode': dominant})
            else:
                seg_len = min(random.uniform(4, 6), duration - t)
                new_segments.append({'start': t, 'end': t + seg_len, 'mode': alt})
            t += seg_len
            use_dominant = not use_dominant
        
        # Preserve face tracking data from original by mapping avg_x
        for seg in new_segments:
            seg_frames = [f for f in frame_data if seg['start'] <= f['timestamp'] <= seg['end']]
            seg['avg_x'] = sum(f['primary_x'] for f in seg_frames) / len(seg_frames) if seg_frames else 0.5
            seg['avg_face_area'] = sum(f['face_area'] for f in seg_frames) / len(seg_frames) if seg_frames else 0
        segments = new_segments
    
    for seg in segments:
        seg_frames = [f for f in frame_data if seg['start'] <= f['timestamp'] <= seg['end']]
        seg['avg_x'] = sum(f['primary_x'] for f in seg_frames) / len(seg_frames) if seg_frames else 0.5
        seg['avg_face_area'] = sum(f['face_area'] for f in seg_frames) / len(seg_frames) if seg_frames else 0
    
    print(f"  [1/4] Found {len(segments)} segments ({total_frames} frames analyzed)", flush=True)
    for i, seg in enumerate(segments):
        print(f"    Seg {i}: {seg['start']:.1f}s → {seg['end']:.1f}s [{seg['mode']}]", flush=True)
    
    return {
        'video_width': w, 'video_height': h,
        'segments': segments,
        'adjusted_start_time': start_time,
        '_frame_data': frame_data
    }

# ============================================
# WHISPER TRANSCRIPTION
# ============================================
def transcribe_audio(video_path, start_time, duration):
    """Transcribe audio using Whisper"""
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
        
        # Cleanup
        del model, result
        gc.collect()
        return words
    finally:
        if os.path.exists(temp.name): os.unlink(temp.name)

# ============================================
# SFX EVENT DETECTION
# ============================================
def get_sfx_events(words):
    """Find keyword-triggered sound effect events"""
    events = []
    for w in words:
        k, v = fuzzy_keyword_match(w['word'], SFX_MAP)
        if k:
            sfx_path = str(ASSETS_DIR / v)
            if os.path.exists(sfx_path):
                events.append({'file': sfx_path, 'time': w['start'], 'keyword': k})
    return events

# ============================================
# FFMPEG FILTER GENERATION
# ============================================
# NOTE: build_zoom_punch_filter removed - zoompan is too CPU-intensive
# and causes crashes on headless servers without GPU.

def generate_crop_filter(analysis, duration, words, tracking=True):
    """Generate the complete FFmpeg filter_complex string"""
    vid_w, vid_h = analysis['video_width'], analysis['video_height']
    segments = analysis['segments']
    if not segments:
        return f"nullsink", "[0:a]atempo=1.05"
    
    out_w, out_h = 1080, 1920
    aspect_9_16 = 9 / 16
    target_fps = 30
    fade_duration = 0.3 if len(segments) > 1 else 0
    
    # Global settings
    speed_factor = 1.05
    pts_expr = f"setpts=PTS/{speed_factor}"
    adjusted_duration = duration / speed_factor
    
    filter_parts, v_names, a_names = [], [], []
    
    # NOTE: Flash/Shake/ZoomPunch effects removed to prevent crashes on headless servers.
    # These effects used complex FFmpeg expressions (nested if(), random() in crop,
    # zoompan) that caused crashes due to expression overflow and CPU overload.
    
    for i, seg in enumerate(segments):
        start, end, mode = seg['start'], seg['end'], seg['mode']
        v_seg, a_seg = f"v{i}", f"a{i}"
        
        # Trim filters (video + audio)
        trim_v = f"fps={target_fps},trim=start={start}:end={end},setpts=PTS-STARTPTS,{pts_expr}"
        trim_a = f"aresample=async=1,atrim=start={start}:end={end},asetpts=PTS-STARTPTS,atempo={speed_factor}"
        
        if mode == 'wide':
            # ═══════════════════════════════════════
            # WIDE MODE: Split screen (top + bottom)
            # ═══════════════════════════════════════
            panel_aspect = out_w / 956
            cw = (min(vid_w // 2 - 20, int(vid_h * panel_aspect)) // 4) * 4
            ch = (int(cw / panel_aspect) // 4) * 4
            lx = max(0, int(vid_w * 0.25) - cw // 2)
            rx = min(vid_w - cw, int(vid_w * 0.75) - cw // 2)
            y = (vid_h - ch) // 2
            panel_h_adj, div_h = 956, 8
            
            part = (
                f"[0:v]{trim_v},split=2[t{i}][b{i}];"
                f"[t{i}]crop={cw}:{ch}:{lx}:{y},scale={out_w}:{panel_h_adj}:flags=lanczos,setsar=1,fps={target_fps},format=yuv420p[top{i}];"
                f"[b{i}]crop={cw}:{ch}:{rx}:{y},scale={out_w}:{panel_h_adj}:flags=lanczos,setsar=1,fps={target_fps},format=yuv420p[bot{i}];"
                f"[top{i}]pad={out_w}:{panel_h_adj + div_h}:0:0:black[tpad{i}];"
                f"[tpad{i}][bot{i}]vstack=inputs=2[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        else:
            # ═══════════════════════════════════════
            # CLOSEUP MODE: Face tracking + zoom punch
            # ═══════════════════════════════════════
            target_h, target_w = vid_h, int(vid_h * aspect_9_16)
            cw = (target_w // 2) * 2  # ensure even
            ch = (target_h // 2) * 2  # ensure even
            
            # Face tracking X position
            seg_frames = [f for f in analysis.get('_frame_data', []) if start <= f['timestamp'] <= end]
            if not tracking or not seg_frames:
                xb = max(0, min(int(seg['avg_x'] * vid_w) - target_w // 2, vid_w - target_w))
                x_expr = str(xb)
            else:
                tracking_points, last_t = [], -1
                for f in seg_frames:
                    if f['timestamp'] >= last_t + 0.4:
                        tracking_points.append((f['timestamp'], f['smoothed_x']))
                        last_t = f['timestamp']
                
                if len(tracking_points) < 2:
                    xb = max(0, min(int(seg['avg_x'] * vid_w) - target_w // 2, vid_w - target_w))
                    x_expr = str(xb)
                else:
                    parts = []
                    first_x = max(0, min(int(tracking_points[0][1] * vid_w) - target_w // 2, vid_w - target_w))
                    for j in range(len(tracking_points) - 1):
                        t1 = tracking_points[j][0]
                        x1 = tracking_points[j][1]
                        t2 = tracking_points[j + 1][0]
                        x2 = tracking_points[j + 1][1]
                        p1 = max(0, min(int(x1 * vid_w) - target_w // 2, vid_w - target_w))
                        p2 = max(0, min(int(x2 * vid_w) - target_w // 2, vid_w - target_w))
                        rel_t1 = round(t1 - start, 3)
                        rel_t2 = round(t2 - start, 3)
                        dur_t = round(t2 - t1, 3)
                        lerp = str(p1) if p1 == p2 else f"({p1}+({p2}-{p1})*(t-{rel_t1})/{dur_t})"
                        if j == len(tracking_points) - 2:
                            parts.append(f"({lerp}*between(t,{rel_t1},{rel_t2}))")
                        else:
                            parts.append(f"({lerp}*gte(t,{rel_t1})*lt(t,{rel_t2}))")
                    first_rel_t = round(tracking_points[0][0] - start, 3)
                    x_expr = f"trunc({first_x}*lt(t,{first_rel_t}) + {' + '.join(parts)})"

            final_x = f"min(max(0,{x_expr}),iw-{cw})"
            
            part = (
                f"[0:v]{trim_v},"
                f"crop=w={cw}:h={ch}:x='{final_x}':y='(ih-oh)/2',"
                f"scale={out_w}:{out_h}:flags=lanczos,"
                f"setsar=1,format=yuv420p[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        
        filter_parts.append(part)
        v_names.append(f"[{v_seg}]")
        a_names.append(f"[{a_seg}]")
    
    # Join all segments using concat (Hard Cuts - Better for virality)
    f_str = ";".join(filter_parts)
    
    if len(segments) == 1:
        # Single segment — direct rename
        f_str = f_str.replace("[v0]", "[stacked]").replace("[a0]", "[main_audio]")
    else:
        # Multiple segments — Concatenate with hard cuts
        v_concat = "".join(v_names)
        a_concat = "".join(a_names)
        f_str += f";{v_concat}concat=n={len(segments)}:v=1:a=0[stacked]"
        f_str += f";{a_concat}concat=n={len(segments)}:v=0:a=1[main_audio]"
    
    # Post-processing chain: color grading → progress bar
    # (Flash and Shake effects removed - caused crashes on headless servers)
    f_str += (
        f";[stacked]eq=contrast=1.15:saturation=1.2:brightness=-0.04[graded_base]"
        f";color=c=yellow@0.9:s=1080x30[pbar]"
        f";[graded_base][pbar]overlay=x='-w+(w*t/{adjusted_duration:.2f})':y=H-30:shortest=1[graded]"
    )
    
    return f_str, "[main_audio]"

# ============================================
# ASS SUBTITLE GENERATION
# ============================================
def format_ass_time(s):
    """Format seconds to ASS time format H:MM:SS.cc"""
    s = max(0, s)
    return f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}.{int((s % 1) * 100):02d}"


def generate_ass_subtitle(words, analysis, title=None, speed_factor=1.0):
    """Generate ASS subtitle with all text effects"""
    if not words:
        return ""
    
    # Adjust timing for speed factor
    if speed_factor != 1.0:
        for w in words:
            w['start'] = w['start'] / speed_factor
            w['end'] = w['end'] / speed_factor
    
    segments = analysis['segments']
    
    # Build word phrases (2 words per subtitle or split on punctuation)
    phrases = []
    cur_phrase, cur_start = [], None
    for w in words:
        if cur_start is None:
            cur_start = w['start']
        cur_phrase.append(w['word'].upper())
        if len(cur_phrase) >= 2 or w['word'].rstrip().endswith(('.', '?', '!', ',')):
            phrases.append({'words': list(cur_phrase), 'start': cur_start, 'end': w['end']})
            cur_phrase, cur_start = [], None
    if cur_phrase:
        phrases.append({'words': list(cur_phrase), 'start': cur_start, 'end': words[-1]['end']})
    
    # ASS Header with proper fonts
    # Uses DejaVu Sans (available in container) - Bold flag (-1) makes it bold
    # Font sizes: Default=80 on 1080x1920 PlayRes = visible bold text
    # Alignment: 2=bottom-center, 5=middle-center, 8=top-center
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
        # Default: white bold, thick black outline, bottom-center (Alignment=2)
        "Style: Default,DejaVu Sans,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,2,0,1,5,3,2,50,50,150,1\n"
        # Highlight: yellow bold, same outline
        "Style: Highlight,DejaVu Sans,80,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,2,0,1,5,3,2,50,50,150,1\n"
        # Hook: large yellow, thick border, middle-center (Alignment=5), MarginV=300 from center
        "Style: Hook,DejaVu Sans,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,4,0,1,6,0,5,60,60,100,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    events = ""
    
    # Hook title with pop animation (first 2.5 seconds)
    # Positioned at top-quarter of screen (\pos(540,350)) to avoid overlapping subtitles
    if title:
        pop = (
            r"{\pos(540,350)"
            r"\fscx100\fscy100"
            r"\t(0,200,\fscx115\fscy115)"      # zoom in
            r"\t(200,400,\fscx100\fscy100)"      # zoom out (bounce)
            r"\be3"                               # edge blur
            r"\bord8"                             # thick border
            r"\3c&H000000&}"                      # border color black
        )
        events += f"Dialogue: 1,0:00:00.00,0:00:02.50,Hook,,0,0,0,,{pop}{title.upper()}\n"
    
    # Subtitle phrases with effects
    for p in phrases:
        # Determine which segment this phrase belongs to (for positioning)
        mode = 'closeup'
        for s in segments:
            if s['start'] <= p['start'] * speed_factor <= s['end']:
                mode = s['mode']
                break
        
        # Position: wide mode = between panels (gap area), closeup = lower third
        # Wide: y=960 = exact center (panel gap). Closeup: y=1550 = lower third
        y = 960 if mode == 'wide' else 1550
        
        # Style each word (highlights + emoji)
        styled_words = []
        for word in p['words']:
            clean_w = word.strip('.,!?;:\"\'-').upper()
            
            # Fuzzy match for emoji and highlight
            k_emoji, emoji = fuzzy_keyword_match(word, EMOJI_MAP)
            
            # Highlight: keyword match OR 15% random chance
            is_highlight = False
            if clean_w in HIGHLIGHT_KEYWORDS:
                is_highlight = True
            else:
                # Check if it contains any of the highlight keywords
                for h in HIGHLIGHT_KEYWORDS:
                    if h in clean_w and len(h) >= 4:
                        is_highlight = True
                        break
            
            if is_highlight or random.random() < 0.15:
                # Use color tag (\c) instead of reset (\r) to avoid breaking position/animation
                styled_words.append(
                    "{\\c&H00FFFF&}" + word +
                    (f" {emoji}" if emoji else "") +
                    "{\\c&HFFFFFF&}"
                )
            else:
                styled_words.append(word + (f" {emoji}" if emoji else ""))
        
        # Pop-in animation for each subtitle line
        pop_tag = r"\fscx120\fscy120\t(0,120,\fscx100\fscy100)"
        text = ' '.join(styled_words)
        
        events += (
            f"Dialogue: 0,{format_ass_time(p['start'])},{format_ass_time(p['end'])},"
            f"Default,,0,0,0,,{{{pop_tag}\\pos(540,{y})}}{text}\n"
        )
    
    return header + events

# ============================================
# MAIN VIDEO PROCESSING
# ============================================
def process_video(input_file, output_file, start_time, duration, ass_file=None, hook_title=None, ai_mode='auto', tracking=True):
    """Main processing function - Optimized for n8n Flow"""
    print(f"\n{'='*50}", flush=True)
    print(f"  VIDEO PROCESSING START", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  Input:    {input_file}", flush=True)
    print(f"  Output:   {output_file}", flush=True)
    print(f"  Start:    {start_time}s, Duration: {duration}s", flush=True)
    print(f"  AI Mode:  {ai_mode}", flush=True)
    print(f"  Tracking: {tracking}", flush=True)
    print(f"  Title:    {hook_title or '(none)'}", flush=True)

    # Check input file
    if not os.path.exists(input_file):
        print(f"  ✗ Error: Input file not found: {input_file}", flush=True)
        sys.exit(1)

    # ── Step 1: AI Video Analysis ──
    analysis = analyze_video_dynamic(input_file, start_time, duration, ai_mode)
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
    sfx_events = get_sfx_events(words)
    print(f"    SFX triggers: {len(sfx_events)}", flush=True)
    for sfx in sfx_events:
        print(f"      → {sfx['keyword']} at {sfx['time']:.1f}s ({os.path.basename(sfx['file'])})", flush=True)
    
    # Generate filter_complex
    f_complex, a_stream = generate_crop_filter(analysis, duration, words, tracking=tracking)
    v_stream = "[graded]"
    speed_factor = 1.05
    
    # ASS subtitle overlay
    if ass_file:
        ass_content = generate_ass_subtitle(words, analysis, title=hook_title, speed_factor=speed_factor)
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        print(f"    ASS subtitle: {ass_file} ({len(ass_content)} bytes)", flush=True)
        
        # Escape path for FFmpeg ass filter
        ass_escaped = ass_file.replace('\\', '/').replace("'", "\\'").replace(":", "\\:")
        f_complex += f";[graded]ass='{ass_escaped}'[outv]"
        v_stream = "[outv]"
    
    # SFX audio mixing
    sfx_filter, sfx_inputs = "", []
    if sfx_events:
        for i, sfx in enumerate(sfx_events):
            rel_t = max(0, sfx['time'] * 1000 / speed_factor)  # ms, adjusted for speed
            sfx_inputs.extend(['-i', sfx['file']])
            sfx_filter += f"[{i + 1}:a]adelay={rel_t:.0f}|{rel_t:.0f}[sfx{i}];"
        
        sfx_refs = ''.join([f'[sfx{i}]' for i in range(len(sfx_events))])
        sfx_filter += f"{a_stream}{sfx_refs}amix=inputs={len(sfx_events) + 1}:normalize=0[mixed_audio]"
        a_stream = "[mixed_audio]"
    
    full_filter = f_complex + (";" + sfx_filter if sfx_filter else "")
    
    # Debug: Print filter_complex
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
        ] + sfx_inputs + [
            '-filter_complex', full_filter,
            '-map', v_stream,
            '-map', a_stream,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
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
        # Final cleanup of any potential artifacts in WORK_DIR
        # Only clean up files specifically related to this run if possible, 
        # but here we'll just ensure gc is called for memory safety.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ============================================
# CLI ENTRY POINT
# ============================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI-Powered Face Detection Auto-Crop - Performance Mode')
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
    
    args = parser.parse_args()
    
    # Merge title arguments
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
        tracking=args.tracking
    )
    
    print(f"\n{'='*50}", flush=True)
    print(f"  PROCESSING COMPLETE!", flush=True)
    print(f"{'='*50}", flush=True)
