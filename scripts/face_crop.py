#!/usr/bin/env python3
"""
Face Detection Auto-Crop for Podcast Videos
Optimized for Tencent Cloud Lighthouse 2GB/2CPU
AI Dynamic Closeup/Split Detection
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json
import wave

# Suppress logging dan force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# OPTIMIZED: Utilizing 4 threads for 8-core CPU
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

try:
    import cv2
    import mediapipe as mp
    import whisper
    import torch
    import gc
    import numpy as np
    
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
except ImportError as e:
    print(f"Error: {e}")
    print("Install: pip install opencv-python mediapipe openai-whisper torch numpy")
    sys.exit(1)

# Working directory
WORK_DIR = '/tmp/clipper'
os.makedirs(WORK_DIR, exist_ok=True)

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
    """Extract video segment dengan h264 encoding"""
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
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        return temp.name
    
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

def analyze_video_dynamic(video_path, start_time, duration, ai_mode='auto'):
    """
    AI-powered dynamic video analysis with SMART CAMERA DETECTION
    - Detects close-up shots based on face SIZE (area), not just count
    - If face is large enough (>8% of frame), treat as close-up
    
    ai_mode: 'auto', 'prefer_closeup', 'prefer_split'
    """
    print(f"  Analyzing video (mode: {ai_mode})...")
    
    detector = FaceDetector(min_confidence=0.3)
    w, h, total_dur = get_video_info(video_path)
    
    if w is None:
        return None
    
    is_ultrawide = (w / h) > 2.0
    
    # Extract segment
    temp_seg = extract_segment_h264(video_path, start_time, duration)
    if not temp_seg:
        return None
    
    # Analyze frames
    cap = cv2.VideoCapture(temp_seg)
    frame_data = []
    frame_count = 0
    
    dual_count = 0
    left_count = 0
    right_count = 0
    closeup_count = 0  # Count frames where face is large (close-up shot)
    total_frames = 0
    
    # Thresholds for smart detection
    CLOSEUP_FACE_AREA_THRESHOLD = 0.08  # 8% of frame = close-up shot
    MEDIUM_FACE_AREA_THRESHOLD = 0.03   # 3% of frame = medium shot
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every 3rd frame untuk save memory
        if frame_count % 3 != 0:
            continue
        
        total_frames += 1
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        faces = detector.detect_faces(frame)
        
        # Count face positions
        if len(faces) >= 2:
            dual_count += 1
        elif len(faces) == 1:
            if faces[0]['center_x'] < 0.4:
                left_count += 1
            elif faces[0]['center_x'] > 0.6:
                right_count += 1
        
        primary = faces[0] if faces else {
            'center_x': 0.5, 'center_y': 0.5, 
            'rel_w': 0, 'rel_h': 0
        }
        
        # Calculate face area (relative to frame)
        face_area = primary['rel_w'] * primary['rel_h']
        is_closeup_shot = face_area >= CLOSEUP_FACE_AREA_THRESHOLD
        is_medium_shot = face_area >= MEDIUM_FACE_AREA_THRESHOLD
        
        if is_closeup_shot:
            closeup_count += 1
        
        # AI Decision Logic with SMART CAMERA DETECTION
        mode = 'closeup'  # Default
        
        if ai_mode == 'prefer_split':
            # Split when 2+ faces, BUT NOT if primary face is already close-up
            if len(faces) >= 2 and not is_closeup_shot:
                mode = 'wide'
            else:
                mode = 'closeup'
        
        elif ai_mode == 'prefer_closeup':
            # Closeup unless clearly needs split
            if len(faces) >= 2 and dual_count > total_frames * 0.3 and not is_closeup_shot:
                mode = 'wide'
            else:
                mode = 'closeup'
        
        else:  # auto mode - SMART DETECTION
            # PRIORITY 1: If face is large (close-up shot), NEVER split
            if is_closeup_shot:
                mode = 'closeup'
            # PRIORITY 2: If 2+ faces and not close-up, split
            elif len(faces) >= 2:
                mode = 'wide'
            # PRIORITY 3: If ultrawide with no/small face, split
            elif len(faces) == 0 and is_ultrawide:
                mode = 'wide'
            elif is_ultrawide and not is_medium_shot:
                mode = 'wide'
            else:
                mode = 'closeup'
        
        frame_data.append({
            'timestamp': ts,
            'face_count': len(faces),
            'primary_x': primary['center_x'],
            'face_area': face_area,
            'is_closeup_shot': is_closeup_shot,
            'mode': mode
        })
        
        del frame
    
    # Smooth primary_x coordinates for better follow-camera effect
    x_coords = [f['primary_x'] for f in frame_data]
    smoothed_x = smooth_coordinates(x_coords, window_size=11)
    for i, f in enumerate(frame_data):
        f['smoothed_x'] = smoothed_x[i]

    cap.release()
    os.unlink(temp_seg)
    detector.close()
    cv2.destroyAllWindows()
    gc.collect()
    
    # Statistics
    dual_ratio = dual_count / total_frames if total_frames > 0 else 0
    closeup_ratio = closeup_count / total_frames if total_frames > 0 else 0
    
    print(f"  Analysis: {total_frames} frames")
    print(f"  Dual faces: {dual_count} ({dual_ratio*100:.1f}%)")
    print(f"  Close-up shots: {closeup_count} ({closeup_ratio*100:.1f}%)")
    print(f"  Left={left_count}, Right={right_count}")
    
    # Build segments with smoothing
    segments = []
    if frame_data:
        curr_mode = frame_data[0]['mode']
        start_t = 0
        min_duration = 2.0  # Minimum 2 detik per segment
        
        for i in range(1, len(frame_data)):
            seg_duration = frame_data[i]['timestamp'] - start_t
            
            # Switch jika mode berubah dan sudah minimal duration
            if frame_data[i]['mode'] != curr_mode and seg_duration >= min_duration:
                segments.append({
                    'start': start_t,
                    'end': frame_data[i]['timestamp'],
                    'mode': curr_mode
                })
                start_t = frame_data[i]['timestamp']
                curr_mode = frame_data[i]['mode']
        
        # Last segment
        segments.append({
            'start': start_t,
            'end': duration,
            'mode': curr_mode
        })
    
    if not segments:
        default_mode = 'wide' if ai_mode == 'prefer_split' else 'closeup'
        segments.append({
            'start': 0,
            'end': duration,
            'mode': default_mode,
            'avg_x': 0.5
        })
    
    # Calculate avg_x and avg_face_area for each segment
    for seg in segments:
        seg_frames = [f for f in frame_data if seg['start'] <= f['timestamp'] <= seg['end']]
        seg['avg_x'] = sum(f['primary_x'] for f in seg_frames) / len(seg_frames) if seg_frames else 0.5
        seg['avg_face_area'] = sum(f['face_area'] for f in seg_frames) / len(seg_frames) if seg_frames else 0
    
    print(f"  Generated {len(segments)} segments:")
    for i, seg in enumerate(segments):
        area_pct = seg.get('avg_face_area', 0) * 100
        print(f"    {i+1}. {seg['start']:.1f}s-{seg['end']:.1f}s => {seg['mode'].upper()} (face area: {area_pct:.1f}%)")
    
    return {
        'video_width': w,
        'video_height': h,
        'segments': segments,
        'adjusted_start_time': start_time,
        '_frame_data': frame_data,  # Keep for tracking
        'stats': {
            'dual_count': dual_count,
            'dual_ratio': dual_ratio,
            'closeup_count': closeup_count,
            'closeup_ratio': closeup_ratio,
            'total_frames': total_frames
        }
    }

def transcribe_audio(video_path, start_time, duration):
    """Transcribe audio menggunakan Whisper tiny model"""
    print("  Transcribing audio...")
    
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=WORK_DIR)
    temp.close()
    
    try:
        # Extract audio
        cmd = [
            'ffmpeg', '-y', '-threads', '1',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            temp.name
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Transcribe dengan turbo model (large-v3-turbo, sangat akurat dan cepat)
        model = whisper.load_model('turbo')

        result = model.transcribe(temp.name, word_timestamps=True, language='id', fp16=False)
        
        words = []
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
        
        # Cleanup
        del model, result
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return words
        
    finally:
        if os.path.exists(temp.name):
            os.unlink(temp.name)

def generate_crop_filter(analysis, duration, words, tracking=True):
    """Generate FFmpeg filter_complex untuk crop dan zoom"""
    w, h = analysis['video_width'], analysis['video_height']
    segments = analysis['segments']
    
    out_w, out_h, panel_h = 1080, 1920, 960
    aspect_9_16 = 9/16
    aspect_panel = 1080/960
    
    target_fps = 30
    fade_duration = 0.3 if len(segments) > 1 else 0
    
    filter_parts = []
    v_names = []
    a_names = []
    
    for i, seg in enumerate(segments):
        start, end, mode = seg['start'], seg['end'], seg['mode']
        v_seg, a_seg = f"v{i}", f"a{i}"
        
        trim_v = f"fps={target_fps},trim=start={start}:end={end},setpts=PTS-STARTPTS"
        trim_a = f"aresample=async=1,atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        
        # Simple zoom (hook zoom aja, skip emphasis untuk save CPU)
        z_expr = "1.0+0.2*between(time,0,1.5)*sin(PI*time/1.5)"
        
        zoom_filter = f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={out_w}x{out_h}:fps={target_fps}"
        zoom_filter_panel = f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={out_w}x{panel_h}:fps={target_fps}"
        
        if mode == 'wide':
            # Split dual panel
            cw = (min(w // 2 - 20, int(h * aspect_panel)) // 2) * 2
            ch = (int(cw / aspect_panel) // 2) * 2
            lx = max(0, int(w*0.25)-cw//2)
            rx = min(w-cw, int(w*0.75)-cw//2)
            y = (h - ch) // 2
            
            part = (
                f"[0:v]{trim_v},split=2[t{i}][b{i}];"
                f"[t{i}]crop={cw}:{ch}:{lx}:{y},scale={out_w}:{panel_h},setsar=1,fps={target_fps},format=yuv420p,{zoom_filter_panel}[top{i}];"
                f"[b{i}]crop={cw}:{ch}:{rx}:{y},scale={out_w}:{panel_h},setsar=1,fps={target_fps},format=yuv420p,{zoom_filter_panel}[bot{i}];"
                f"[top{i}][bot{i}]vstack=inputs=2[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        else:
            # Closeup single person
            ch = (h // 2) * 2
            cw = (int(h * aspect_9_16) // 2) * 2
            
            # SPEAKER TRACKING LOGIC
            seg_frames = [f for f in analysis.get('_frame_data', []) if start <= f['timestamp'] <= end]
            
            if not tracking or not seg_frames:
                xb = max(0, min(int(seg['avg_x'] * w) - cw // 2, w - cw))
                x_expr = str(xb)
            else:
                # Sub-sample tracking points to avoid too long expression (every 0.4s)
                tracking_points = []
                last_t = -1
                for f in seg_frames:
                    if f['timestamp'] >= last_t + 0.4:
                        tracking_points.append((f['timestamp'], f['smoothed_x']))
                        last_t = f['timestamp']
                
                if len(tracking_points) < 2:
                    # Fallback to static avg_x
                    xb = max(0, min(int(seg['avg_x'] * w) - cw // 2, w - cw))
                    x_expr = str(xb)
                else:
                    # Create FLAT expression for linear movement to avoid FFmpeg recursion limit
                    # Formula: trunc(sum((p1 + (p2-p1)*(t-t1)/(t2-t1)) * between(t, t1, t2)))
                    parts = []
                    
                    # Ensure first value is covered if start > first point (unlikely with current logic but safe)
                    first_x = max(0, min(int(tracking_points[0][1] * w) - cw // 2, w - cw))
                    
                    for j in range(len(tracking_points) - 1):
                        t1, x1 = tracking_points[j]
                        t2, x2 = tracking_points[j+1]
                        
                        # Convert rel to px crop x
                        p1 = max(0, min(int(x1 * w) - cw // 2, w - cw))
                        p2 = max(0, min(int(x2 * w) - cw // 2, w - cw))
                        
                        rel_t1 = round(t1 - start, 3)
                        rel_t2 = round(t2 - start, 3)
                        dur = round(t2 - t1, 3)
                        
                        if p1 == p2:
                            lerp = str(p1)
                        else:
                            lerp = f"({p1}+({p2}-{p1})*(t-{rel_t1})/{dur})"
                        
                        # Use gte/lt to avoid overlap doubling if simply summing
                        # The last segment will use between to be inclusive
                        if j == len(tracking_points) - 2:
                            parts.append(f"({lerp}*between(t,{rel_t1},{rel_t2}))")
                        else:
                            parts.append(f"({lerp}*gte(t,{rel_t1})*lt(t,{rel_t2}))")
                    
                    # Full expression: trunc(part1 + part2 + ...)
                    # Add a fallback for t < tracking_points[0]
                    first_rel_t = round(tracking_points[0][0] - start, 3)
                    x_expr = f"trunc({first_x}*lt(t,{first_rel_t}) + {' + '.join(parts)})"

            part = (
                f"[0:v]{trim_v},crop={cw}:{ch}:'{x_expr}':0,"
                f"scale={out_w}:{out_h},setsar=1,fps={target_fps},format=yuv420p,{zoom_filter}[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        
        filter_parts.append(part)
        v_names.append(f"[{v_seg}]")
        a_names.append(f"[{a_seg}]")
    
    if not segments or not v_names:
        return f"[0:v]scale={out_w}:{out_h},setsar=1,fps={target_fps},format=yuv420p[stacked];[0:a]acopy[stackeda]", "[stackeda]"
    
    f_str = ";".join(filter_parts)
    
    # Concatenate segments
    if len(segments) == 1:
        f_str = f_str.replace("[v0]", "[stacked]").replace("[a0]", "[main_audio]")
    else:
        # Multiple segments dengan crossfade
        cv, ca = v_names[0], a_names[0]
        chains = []
        
        cum_t = segments[0]['end'] - segments[0]['start']
        
        for i in range(1, len(segments)):
            vo = f"[vx{i}]" if i < len(segments)-1 else "[stacked]"
            ao = f"[ax{i}]" if i < len(segments)-1 else "[main_audio]"
            
            off = max(0, cum_t - fade_duration)
            
            chains.append(f"{cv}{v_names[i]}xfade=transition=dissolve:duration={fade_duration}:offset={off:.3f}{vo}")
            chains.append(f"{ca}{a_names[i]}acrossfade=duration={fade_duration}{ao}")
            
            cum_t = off + (segments[i]['end'] - segments[i]['start'])
            cv, ca = vo, ao
        
        f_str += ";" + ";".join(chains)
    
    # Adjust subtitle timestamps
    ts, cur = 0, 0
    for wd in words:
        while cur < len(segments)-1 and wd['start'] > segments[cur]['end']:
            cur += 1
            ts += fade_duration
        wd['start'] = max(0, wd['start'] - ts)
        wd['end'] = max(wd['start'] + 0.1, wd['end'] - ts)
    
    return f_str, "[main_audio]"

def format_ass_time(s):
    """Format time untuk ASS subtitle"""
    return f"{int(s//3600)}:{int((s%3600)//60):02d}:{int(s%60):02d}.{int((s%1)*100):02d}"

def generate_ass_subtitle(words, analysis, title=None):
    """Generate ASS subtitle file with READABLE font"""
    if not words:
        return ""
    
    segments = analysis['segments']
    phrases = []
    cur_phrase = []
    cur_start = None
    
    # Group words into phrases
    for w in words:
        if cur_start is None:
            cur_start = w['start']
        
        cur_phrase.append(w['word'])
        
        # End phrase on 4 words or punctuation
        if len(cur_phrase) >= 4 or w['word'].endswith(('.', '?', '!', ',')):
            phrases.append({
                'text': ' '.join(cur_phrase),
                'start': cur_start,
                'end': w['end']
            })
            cur_phrase = []
            cur_start = None
    
    # Remaining words
    if cur_phrase:
        phrases.append({
            'text': ' '.join(cur_phrase),
            'start': cur_start,
            'end': words[-1]['end']
        })
    
    # ASS header with IMPROVED READABLE FONTS
    # Using Arial Bold (universally available) with:
    # - Larger font size (70 for subtitle, 100 for hook)
    # - Soft white color with dark outline
    # - Good shadow for visibility
    # - Rounded corners via BorderStyle 3
    header = (
        "[Script Info]\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Default subtitle style - Clean, readable, eye-friendly
        # Font: Arial Bold, Size: 70, White text with dark gray outline
        # BackColour: Semi-transparent black box for readability
        "Style: Default,Arial,70,&H00FFFFFF,&H00FFFFFF,&H00222222,&H80000000,-1,0,0,0,100,100,1,0,1,4,2,2,50,50,150,1\n"
        # Hook style - Bold yellow for attention
        "Style: Hook,Arial,100,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,3,6,3,5,60,60,200,1\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    events = ""
    
    # Title hook (first 3 seconds)
    if title:
        events += f"Dialogue: 1,0:00:00.00,0:00:03.00,Hook,,0,0,0,,{{\\fad(200,200)\\b1}}{title.upper()}\n"
    
    # Subtitle events
    for p in phrases:
        # Determine mode for vertical position
        mode = 'closeup'
        for s in segments:
            if s['start'] <= p['start'] <= s['end']:
                mode = s['mode']
                break
        
        # Position and font size based on mode
        # Wide (split): center between panels at y=960
        # Closeup: near bottom at y=1750
        y = 960 if mode == 'wide' else 1750
        fs = 60 if mode == 'wide' else 70
        
        # Clean subtitle with fade effect and proper formatting
        events += f"Dialogue: 0,{format_ass_time(p['start'])},{format_ass_time(p['end'])},Default,,0,0,0,,{{\\fs{fs}\\pos(540,{y})\\fad(80,80)\\bord4}}{p['text']}\n"
    
    return header + events

def process_video(input_file, output_file, start_time, duration, ass_file=None, hook_title=None, ai_mode='auto', tracking=True):
    """Main processing function"""
    print(f"\n=== Video Processing Start ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Start: {start_time}s, Duration: {duration}s")
    print(f"AI Mode: {ai_mode}")
    
    # Validate ai_mode
    valid_modes = ['auto', 'prefer_closeup', 'prefer_split']
    if ai_mode not in valid_modes:
        print(f"Warning: Invalid ai_mode '{ai_mode}', using 'auto'")
        ai_mode = 'auto'
    
    # Check input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Video analysis
    analysis = analyze_video_dynamic(input_file, start_time, duration, ai_mode)
    if not analysis:
        print("Error: Video analysis failed")
        return
    
    actual_start = analysis['adjusted_start_time']
    
    # Transcription
    words = transcribe_audio(input_file, actual_start, duration)
    
    # Generate filter
    f_complex, a_stream = generate_crop_filter(analysis, duration, words, tracking=tracking)
    
    # Cleanup before FFmpeg
    gc.collect()
    
    # Generate subtitle
    v_stream = "[stacked]"
    if ass_file:
        ass_content = generate_ass_subtitle(words, analysis, title=hook_title)
        
        # Write ASS file
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        
        print(f"  ✓ ASS file created: {ass_file}")
    
    # Cleanup before FFmpeg
    del analysis, words
    gc.collect()
    
    # Build FFmpeg filter with ASS
    if ass_file and os.path.exists(ass_file) and os.path.getsize(ass_file) > 100:
        # Escape ASS path untuk FFmpeg
        ass_escaped = ass_file.replace('\\', '\\\\\\\\').replace(':', '\\\\:')
        f_complex += f";[stacked]ass='{ass_escaped}'[outv]"
        v_stream = "[outv]"
        print(f"  ✓ ASS subtitle enabled")
    else:
        print(f"  ⚠ ASS subtitle skipped")
    
    # FFmpeg command
    cmd = [
        'ffmpeg', '-y', '-threads', '4',
        '-ss', str(actual_start),
        '-t', str(duration),
        '-i', input_file,
        '-filter_complex', f_complex,
        '-map', v_stream,
        '-map', a_stream,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '20',  # Lower CRF for higher quality (~10Mbps target)
        '-c:a', 'aac',
        '-b:a', '256k',
        '-ar', '48000',
        '-ac', '2',
        '-max_muxing_queue_size', '4096',
        '-movflags', '+faststart',
        output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  FFmpeg failed: {result.stderr[-500:]}")
    else:
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024*1024)
            print(f"  ✓ Success: {output_file} ({size_mb:.2f} MB)")
        else:
            print("  Error: Output file not created")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI-Powered Face Detection Auto-Crop (Optimized for 2GB RAM)'
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', required=True, help='Output video file')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('-d', '--duration', type=float, default=30, help='Duration (seconds, max 60 recommended)')
    parser.add_argument('--ass-file', help='Output ASS subtitle file')
    parser.add_argument('-t', '--title', help='Hook title banner')
    parser.add_argument('--ai-mode',
        choices=['auto', 'prefer_closeup', 'prefer_split'],
        default='auto',
        help='AI decision mode: auto (intelligent), prefer_closeup, prefer_split'
    )
    # Alias untuk compatibility dengan n8n
    parser.add_argument('-title', dest='title_alias', help='Hook title banner alias')
    parser.add_argument('--tracking', action='store_true', default=True, help='Enable speaker tracking (follow-camera)')
    parser.add_argument('--no-tracking', action='store_false', dest='tracking', help='Disable speaker tracking')
    
    args = parser.parse_args()
    
    # Merge title arguments
    if args.title_alias and not args.title:
        args.title = args.title_alias
    
    # Validate duration untuk 2GB RAM
    if args.duration > 60:
        print(f"Warning: Duration {args.duration}s may cause OOM on 2GB RAM")
        print("Recommended max: 60 seconds")
    
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
    
    print("\n=== Processing Complete! ===")
