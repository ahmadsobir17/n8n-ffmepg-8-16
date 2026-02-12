#!/usr/bin/env python3

"""
Face Detection Auto-Crop for Podcast Videos - VIRAL EDITION
Optimized for 8-core 16GB RAM VPS
AI Dynamic Closeup/Split Detection + Zoom Punches + SFX + Progress Bar
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json
import wave
import random

# Suppress logging dan force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ✅ FIX: Force disable GPU/OpenGL untuk MediaPipe + OpenCV
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# ✅ OPTIMIZED: Utilizing 6 threads for 8-core CPU (75% utilization)
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
os.makedirs(WORK_DIR, exist_ok=True)

# SFX directory (for sound effects)
SFX_DIR = '/app/sfx'
os.makedirs(SFX_DIR, exist_ok=True)

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
        'ffmpeg', '-y', '-threads', '2',
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
    print(f"  Error extracting segment: {result.stderr}")
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
    """
    print(f"  Analyzing video (mode: {ai_mode})...")
    detector = FaceDetector(min_confidence=0.3)
    
    w, h, total_dur = get_video_info(video_path)
    if w is None:
        return None
    
    is_ultrawide = (w / h) > 2.0
    
    temp_seg = extract_segment_h264(video_path, start_time, duration)
    if not temp_seg:
        return None
    
    cap = cv2.VideoCapture(temp_seg)
    frame_data = []
    frame_count = 0
    dual_count = 0
    left_count = 0
    right_count = 0
    closeup_count = 0
    total_frames = 0
    
    CLOSEUP_FACE_AREA_THRESHOLD = 0.08
    MEDIUM_FACE_AREA_THRESHOLD = 0.03
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        
        total_frames += 1
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        faces = detector.detect_faces(frame)
        
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
        
        face_area = primary['rel_w'] * primary['rel_h']
        is_closeup_shot = face_area >= CLOSEUP_FACE_AREA_THRESHOLD
        is_medium_shot = face_area >= MEDIUM_FACE_AREA_THRESHOLD
        
        if is_closeup_shot:
            closeup_count += 1
        
        mode = 'closeup'
        
        if ai_mode == 'prefer_split':
            if len(faces) >= 2 and not is_closeup_shot:
                mode = 'wide'
            else:
                mode = 'closeup'
        elif ai_mode == 'prefer_closeup':
            if len(faces) >= 2 and dual_count > total_frames * 0.3 and not is_closeup_shot:
                mode = 'wide'
            else:
                mode = 'closeup'
        else:
            if is_closeup_shot:
                mode = 'closeup'
            elif len(faces) >= 2:
                mode = 'wide'
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
    
    x_coords = [f['primary_x'] for f in frame_data]
    smoothed_x = smooth_coordinates(x_coords, window_size=11)
    for i, f in enumerate(frame_data):
        f['smoothed_x'] = smoothed_x[i]
    
    cap.release()
    os.unlink(temp_seg)
    detector.close()
    cv2.destroyAllWindows()
    gc.collect()
    
    dual_ratio = dual_count / total_frames if total_frames > 0 else 0
    closeup_ratio = closeup_count / total_frames if total_frames > 0 else 0
    
    print(f"  Analysis: {total_frames} frames")
    print(f"  Dual faces: {dual_count} ({dual_ratio*100:.1f}%)")
    print(f"  Close-up shots: {closeup_count} ({closeup_ratio*100:.1f}%)")
    print(f"  Left={left_count}, Right={right_count}")
    
    segments = []
    if frame_data:
        curr_mode = frame_data[0]['mode']
        start_t = 0
        min_duration = 2.0
        
        for i in range(1, len(frame_data)):
            seg_duration = frame_data[i]['timestamp'] - start_t
            if frame_data[i]['mode'] != curr_mode and seg_duration >= min_duration:
                segments.append({
                    'start': start_t,
                    'end': frame_data[i]['timestamp'],
                    'mode': curr_mode
                })
                start_t = frame_data[i]['timestamp']
                curr_mode = frame_data[i]['mode']
        
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
        '_frame_data': frame_data,
        'stats': {
            'dual_count': dual_count,
            'dual_ratio': dual_ratio,
            'closeup_count': closeup_count,
            'closeup_ratio': closeup_ratio,
            'total_frames': total_frames
        }
    }

def transcribe_audio(video_path, start_time, duration):
    """Transcribe audio using Whisper small model with optimized params"""
    print("  Transcribing audio...")
    temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=WORK_DIR)
    temp.close()
    
    try:
        cmd = [
            'ffmpeg', '-y', '-threads', '2',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            temp.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
        if result.returncode != 0:
            print(f"  Error extracting audio: {result.stderr}")
            return []
        
        model = whisper.load_model('small')
        result = model.transcribe(
            temp.name,
            word_timestamps=True,
            language='id',
            fp16=False,
            beam_size=5,
            best_of=5,
        )
        
        words = []
        for segment in result.get('segments', []):
            for word in segment.get('words', []):
                words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
        
        for w in words:
            w['start'] = max(0, w['start'] - 0.05)
            w['end'] = w['end'] + 0.05
        
        del model, result
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return words
    
    finally:
        if os.path.exists(temp.name):
            os.unlink(temp.name)

def detect_zoom_moments(words, segments):
    """
    🔥 VIRAL FEATURE 1: Detect moments for zoom punches
    Criteria: Question words, emphasis words, emotional words
    """
    zoom_keywords = {
        'APA', 'KENAPA', 'BAGAIMANA', 'SIAPA', 'DIMANA', 'KAPAN',
        'GAK', 'TIDAK', 'JANGAN', 'HARUS', 'PENTING', 'BAHAYA',
        'AMAZING', 'LUAR', 'BIASA', 'GILA', 'HEBAT', 'PARAH',
        'WOW', 'SERIUS', 'PASTI', 'YAKIN', 'PERCAYA'
    }
    
    zoom_moments = []
    for w in words:
        word_upper = w['word'].upper().strip('.,!?;:"\'-')
        
        is_keyword = word_upper in zoom_keywords
        is_emphasis = len(word_upper) > 7
        
        if is_keyword or (is_emphasis and random.random() < 0.3):
            zoom_moments.append({
                'start': w['start'],
                'end': w['end'],
                'zoom': 1.15 if is_keyword else 1.10,
                'word': w['word']
            })
    
    print(f"  🔍 Detected {len(zoom_moments)} zoom punch moments")
    return zoom_moments

def generate_sfx_timeline(segments, words):
    """
    🔊 VIRAL FEATURE 2: Generate sound effects timeline
    - Whoosh on segment transitions
    - Pop on emphasized captions (every 5th word)
    """
    sfx_events = []
    
    for i in range(len(segments) - 1):
        sfx_events.append({
            'type': 'whoosh',
            'time': segments[i]['end'],
            'file': 'whoosh.mp3'
        })
    
    for i, w in enumerate(words):
        if i % 5 == 0:
            sfx_events.append({
                'type': 'pop',
                'time': w['start'],
                'file': 'pop.mp3'
            })
    
    print(f"  🔊 Generated {len(sfx_events)} SFX events ({len([e for e in sfx_events if e['type']=='whoosh'])} whoosh, {len([e for e in sfx_events if e['type']=='pop'])} pop)")
    return sfx_events

def create_default_sfx():
    """Create silent placeholder SFX if files don't exist"""
    whoosh_path = os.path.join(SFX_DIR, 'whoosh.mp3')
    pop_path = os.path.join(SFX_DIR, 'pop.mp3')
    
    for path in [whoosh_path, pop_path]:
        if not os.path.exists(path):
            cmd = [
                'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
                '-t', '0.3', '-q:a', '9', path
            ]
            subprocess.run(cmd, capture_output=True)
            print(f"  ⚠️ Created silent placeholder: {path}")

def generate_crop_filter(analysis, duration, words, tracking=True, enable_zoom=True, enable_progress=True):
    """
    Generate FFmpeg filter_complex untuk crop, zoom, dan progress bar
    🔥 VIRAL ENHANCEMENTS: Zoom punches + Progress bar
    """
    w, h = analysis['video_width'], analysis['video_height']
    segments = analysis['segments']
    out_w, out_h, panel_h = 1080, 1920, 960
    aspect_9_16 = 9/16
    aspect_panel = 1080/960
    target_fps = 30
    fade_duration = 0.3 if len(segments) > 1 else 0
    
    zoom_moments = detect_zoom_moments(words, segments) if enable_zoom else []
    
    filter_parts = []
    v_names = []
    a_names = []
    
    for i, seg in enumerate(segments):
        start, end, mode = seg['start'], seg['end'], seg['mode']
        v_seg, a_seg = f"v{i}", f"a{i}"
        trim_v = f"fps={target_fps},trim=start={start}:end={end},setpts=PTS-STARTPTS"
        trim_a = f"aresample=async=1,atrim=start={start}:end={end},asetpts=PTS-STARTPTS"
        
        if mode == 'wide':
            cw = (min(w // 2 - 20, int(h * aspect_panel)) // 2) * 2
            ch = (int(cw / aspect_panel) // 2) * 2
            lx = max(0, int(w*0.25)-cw//2)
            rx = min(w-cw, int(w*0.75)-cw//2)
            y = (h - ch) // 2
            
            panel_h_adj = 956
            div_h = 8
            
            part = (
                f"[0:v]{trim_v},split=2[t{i}][b{i}];"
                f"[t{i}]crop={cw}:{ch}:{lx}:{y},scale={out_w}:{panel_h_adj}:flags=lanczos,setsar=1,fps={target_fps},format=yuv420p[top{i}];"
                f"[b{i}]crop={cw}:{ch}:{rx}:{y},scale={out_w}:{panel_h_adj}:flags=lanczos,setsar=1,fps={target_fps},format=yuv420p[bot{i}];"
                f"[top{i}]pad={out_w}:{panel_h_adj + div_h}:0:0:black[tpad{i}];"
                f"[tpad{i}][bot{i}]vstack=inputs=2[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        else:
            ch = (h // 2) * 2
            cw = (int(h * aspect_9_16) // 2) * 2
            
            seg_frames = [f for f in analysis.get('_frame_data', []) if start <= f['timestamp'] <= end]
            if not tracking or not seg_frames:
                xb = max(0, min(int(seg['avg_x'] * w) - cw // 2, w - cw))
                x_expr = str(xb)
            else:
                tracking_points = []
                last_t = -1
                for f in seg_frames:
                    if f['timestamp'] >= last_t + 0.4:
                        tracking_points.append((f['timestamp'], f['smoothed_x']))
                        last_t = f['timestamp']
                
                if len(tracking_points) < 2:
                    xb = max(0, min(int(seg['avg_x'] * w) - cw // 2, w - cw))
                    x_expr = str(xb)
                else:
                    parts = []
                    first_x = max(0, min(int(tracking_points[0][1] * w) - cw // 2, w - cw))
                    
                    for j in range(len(tracking_points) - 1):
                        t1, x1 = tracking_points[j]
                        t2, x2 = tracking_points[j+1]
                        p1 = max(0, min(int(x1 * w) - cw // 2, w - cw))
                        p2 = max(0, min(int(x2 * w) - cw // 2, w - cw))
                        rel_t1 = round(t1 - start, 3)
                        rel_t2 = round(t2 - start, 3)
                        dur = round(t2 - t1, 3)
                        
                        if p1 == p2:
                            lerp = str(p1)
                        else:
                            lerp = f"({p1}+({p2}-{p1})*(t-{rel_t1})/{dur})"
                        
                        if j == len(tracking_points) - 2:
                            parts.append(f"({lerp}*between(t,{rel_t1},{rel_t2}))")
                        else:
                            parts.append(f"({lerp}*gte(t,{rel_t1})*lt(t,{rel_t2}))")
                    
                    first_rel_t = round(tracking_points[0][0] - start, 3)
                    x_expr = f"trunc({first_x}*lt(t,{first_rel_t}) + {' + '.join(parts)})"
            
            part = (
                f"[0:v]{trim_v},crop={cw}:{ch}:'{x_expr}':0,"
                f"scale={out_w}:{out_h}:flags=lanczos,setsar=1,fps={target_fps},format=yuv420p[{v_seg}];"
                f"[0:a]{trim_a}[{a_seg}]"
            )
        
        filter_parts.append(part)
        v_names.append(f"[{v_seg}]")
        a_names.append(f"[{a_seg}]")
    
    if not segments or not v_names:
        return f"[0:v]scale={out_w}:{out_h},setsar=1,fps={target_fps},format=yuv420p[stacked];[0:a]acopy[stackeda]", "[stackeda]", "[stacked]"
    
    f_str = ";".join(filter_parts)
    
    if len(segments) == 1:
        f_str = f_str.replace("[v0]", "[stacked]").replace("[a0]", "[main_audio]")
    else:
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
    
    ts, cur = 0, 0
    for wd in words:
        while cur < len(segments)-1 and wd['start'] > segments[cur]['end']:
            cur += 1
            ts += fade_duration
        wd['start'] = max(0, wd['start'] - ts)
        wd['end'] = max(wd['start'] + 0.1, wd['end'] - ts)
    
    f_str += (
        ";[stacked]vignette=angle=0.3:aspect=9/16[vignetted]"
        ";[vignetted]eq=contrast=1.2:saturation=1.1:brightness=-0.02[graded]"
    )
    
    if enable_zoom and zoom_moments:
        zoom_conditions = []
        for zm in zoom_moments:
            zoom_in_end = zm['start'] + 0.15
            zoom_out_start = zm['end'] - 0.15
            
            zoom_conditions.append(
                f"if(between(t,{zm['start']},{zoom_in_end}),"
                f"1+({zm['zoom']}-1)*(t-{zm['start']})/0.15,"
                f"if(between(t,{zoom_in_end},{zoom_out_start}),{zm['zoom']},"
                f"if(between(t,{zoom_out_start},{zm['end']}),"
                f"{zm['zoom']}-({zm['zoom']}-1)*(t-{zoom_out_start})/0.15,1)))"
            )
        
        if len(zoom_conditions) > 10:
            zoom_conditions = zoom_conditions[:10]
            print(f"  ⚠️ Limited zoom punches to 10 (from {len(zoom_moments)})")
        
        zoom_expr = zoom_conditions[0] if zoom_conditions else "1"
        for cond in zoom_conditions[1:]:
            zoom_expr = cond.replace("1))", f"{zoom_expr}))")
        
        f_str += f";[graded]zoompan=z='{zoom_expr}':d=1:s={out_w}x{out_h}:fps={target_fps}[zoomed]"
        final_video = "[zoomed]"
    else:
        final_video = "[graded]"
    
    if enable_progress:
        f_str += (
            f";{final_video}drawbox="
            f"x=0:y=0:"
            f"w='w*t/{duration}':h=6:"
            f"color=yellow@0.9:t=fill"
            f"[with_progress]"
        )
        final_video = "[with_progress]"
    
    return f_str, "[main_audio]", final_video

HIGHLIGHT_KEYWORDS = {
    'INDONESIA', 'RASULULLAH', 'ISLAM', 'ALLAH', 'QURAN',
    'MUSLIM', 'SUNNAH', 'DAKWAH', 'MUKMIN', 'IMAN',
    'JIHAD', 'HIDAYAH', 'TAQWA', 'IBADAH', 'AKHIRAT',
}

def format_ass_time(s):
    return f"{int(s//3600)}:{int((s%3600)//60):02d}:{int(s%60):02d}.{int((s%1)*100):02d}"

def _apply_highlight(word_upper):
    clean = word_upper.strip('.,!?;:"\'-')
    if clean in HIGHLIGHT_KEYWORDS:
        return True
    if random.random() < 0.15:
        return True
    return False

def generate_ass_subtitle(words, analysis, title=None):
    if not words:
        return ""
    
    segments = analysis['segments']
    phrases = []
    cur_phrase = []
    cur_words = []
    cur_start = None
    
    for w in words:
        if cur_start is None:
            cur_start = w['start']
        cur_phrase.append(w['word'].upper())
        cur_words.append(w)
        
        if len(cur_phrase) >= 2 or w['word'].rstrip().endswith(('.', '?', '!', ',')):
            phrases.append({
                'words': list(cur_phrase),
                'start': cur_start,
                'end': w['end']
            })
            cur_phrase = []
            cur_words = []
            cur_start = None
    
    if cur_phrase:
        phrases.append({
            'words': list(cur_phrase),
            'start': cur_start,
            'end': words[-1]['end']
        })
    
    header = (
        "[Script Info]\\n"
        "ScriptType: v4.00+\\n"
        "PlayResX: 1080\\n"
        "PlayResY: 1920\\n"
        "[V4+ Styles]\\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\\n"
        "Style: Default,Arial Black,85,"
        "&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,2,0,1,6,3,2,50,50,150,1\\n"
        "Style: Highlight,Arial Black,85,"
        "&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,2,0,1,6,3,2,50,50,150,1\\n"
        "Style: Hook,Arial Black,110,"
        "&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,3,0,1,7,4,5,60,60,200,1\\n"
        "[Events]\\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\\n"
    )
    
    events = ""
    
    if title:
        pop = r"{\\fscx120\\fscy120\\t(0,150,\\fscx100\\fscy100)\\fad(200,200)\\b1}"
        events += f"Dialogue: 1,0:00:00.00,0:00:03.00,Hook,,0,0,0,,{pop}{title.upper()}\\n"
    
    for p in phrases:
        mode = 'closeup'
        for s in segments:
            if s['start'] <= p['start'] <= s['end']:
                mode = s['mode']
                break
        
        y = 960 if mode == 'wide' else 1650
        
        styled_words = []
        for word in p['words']:
            if _apply_highlight(word):
                styled_words.append("{\\\\rHighlight}" + word + "{\\\\rDefault}")
            else:
                styled_words.append(word)
        text = ' '.join(styled_words)
        
        pop_tag = r"\\fscx110\\fscy110\\t(0,100,\\fscx100\\fscy100)"
        events += (
            f"Dialogue: 0,{format_ass_time(p['start'])},{format_ass_time(p['end'])},"
            f"Default,,0,0,0,,"
            f"{{{pop_tag}\\\\pos(540,{y})}}{text}\\n"
        )
    
    return header + events

def process_video(input_file, output_file, start_time, duration, ass_file=None, hook_title=None, 
                 ai_mode='auto', tracking=True, enable_zoom=True, enable_sfx=True, enable_progress=True):
    """
    Main processing function
    🔥 VIRAL EDITION with Zoom Punches + SFX + Progress Bar
    """
    print(f"\\n=== Video Processing Start (VIRAL EDITION) ===", flush=True)
    print(f"Input: {input_file}", flush=True)
    print(f"Output: {output_file}", flush=True)
    print(f"Start: {start_time}s, Duration: {duration}s", flush=True)
    print(f"AI Mode: {ai_mode}", flush=True)
    print(f"🔥 Zoom Punches: {'ON' if enable_zoom else 'OFF'}", flush=True)
    print(f"🔊 Sound Effects: {'ON' if enable_sfx else 'OFF'}", flush=True)
    print(f"📊 Progress Bar: {'ON' if enable_progress else 'OFF'}", flush=True)
    
    valid_modes = ['auto', 'prefer_closeup', 'prefer_split']
    if ai_mode not in valid_modes:
        print(f"Warning: Invalid ai_mode '{ai_mode}', using 'auto'", flush=True)
        ai_mode = 'auto'
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    if enable_sfx:
        create_default_sfx()
    
    analysis = analyze_video_dynamic(input_file, start_time, duration, ai_mode)
    if not analysis:
        print("Error: Video analysis failed")
        sys.exit(1)
    
    actual_start = analysis['adjusted_start_time']
    
    words = transcribe_audio(input_file, actual_start, duration)
    
    sfx_events = generate_sfx_timeline(analysis['segments'], words) if enable_sfx else []
    
    f_complex, a_stream, final_video = generate_crop_filter(
        analysis, duration, words, 
        tracking=tracking, 
        enable_zoom=enable_zoom,
        enable_progress=enable_progress
    )
    
    gc.collect()
    
    v_stream = final_video
    if ass_file:
        ass_content = generate_ass_subtitle(words, analysis, title=hook_title)
        
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        print(f"  ✓ ASS file created: {ass_file}")
    
    del analysis, words
    gc.collect()
    
    if ass_file and os.path.exists(ass_file) and os.path.getsize(ass_file) > 100:
        ass_escaped = ass_file.replace('\\\\', '/').replace("'", r"\\'").replace(":", r"\\:")
        f_complex += f";{final_video}ass='{ass_escaped}'[outv]"
        v_stream = "[outv]"
        print(f"  ✓ ASS subtitle enabled")
    else:
        print(f"  ⚠ ASS subtitle skipped")
    
    sfx_inputs = []
    sfx_filters = []
    
    if enable_sfx and sfx_events:
        whoosh_times = [e['time'] for e in sfx_events if e['type'] == 'whoosh']
        pop_times = [e['time'] for e in sfx_events if e['type'] == 'pop']
        
        input_idx = 1
        
        if whoosh_times:
            whoosh_path = os.path.join(SFX_DIR, 'whoosh.mp3')
            for t in whoosh_times[:5]:
                sfx_inputs.extend(['-i', whoosh_path])
                sfx_filters.append(f"[{input_idx}:a]adelay={int(t*1000)}|{int(t*1000)},volume=0.3[sfx{input_idx}]")
                input_idx += 1
        
        if pop_times:
            pop_path = os.path.join(SFX_DIR, 'pop.mp3')
            for t in pop_times[:10]:
                sfx_inputs.extend(['-i', pop_path])
                sfx_filters.append(f"[{input_idx}:a]adelay={int(t*1000)}|{int(t*1000)},volume=0.2[sfx{input_idx}]")
                input_idx += 1
        
        if sfx_filters:
            all_sfx = ''.join([f"[sfx{i}]" for i in range(1, input_idx)])
            f_complex += f";{';'.join(sfx_filters)};[main_audio]{all_sfx}amix=inputs={input_idx}:duration=first[final_audio]"
            a_stream = "[final_audio]"
            print(f"  ✓ SFX enabled: {len(sfx_filters)} effects")
    
    cmd = [
        'ffmpeg', '-y', '-threads', '8',
        '-ss', str(actual_start),
        '-t', str(duration),
        '-i', input_file,
    ]
    
    cmd.extend(sfx_inputs)
    
    cmd.extend([
        '-filter_complex', f_complex,
        '-map', v_stream,
        '-map', a_stream,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '24',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-ar', '48000',
        '-ac', '2',
        '-max_muxing_queue_size', '4096',
        '-movflags', '+faststart',
        output_file
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    
    if result.returncode != 0:
        print(f"--- FFmpeg Command Failed ---")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    else:
        if os.path.exists(output_file):
            size_mb = os.path.getsize(output_file) / (1024*1024)
            print(f"  ✓ Success: {output_file} ({size_mb:.2f} MB)")
            if size_mb < 0.001:
                print("  ⚠ Warning: Output file is effectively empty (0 bytes).")
                sys.exit(1)
        else:
            print("  Error: Output file not created")
            sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI-Powered Face Detection Auto-Crop - VIRAL EDITION (8-core 16GB RAM)'
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input video file')
    parser.add_argument('-o', '--output', required=True, help='Output video file')
    parser.add_argument('-s', '--start', type=float, default=0, help='Start time (seconds)')
    parser.add_argument('-d', '--duration', type=float, default=30, help='Duration (seconds, max 90)')
    parser.add_argument('--ass-file', help='Output ASS subtitle file')
    parser.add_argument('-t', '--title', help='Hook title banner')
    parser.add_argument('--ai-mode',
        choices=['auto', 'prefer_closeup', 'prefer_split'],
        default='auto',
        help='AI decision mode'
    )
    
    parser.add_argument('-title', dest='title_alias', help='Hook title banner alias')
    parser.add_argument('--tracking', action='store_true', default=True, help='Enable speaker tracking')
    parser.add_argument('--no-tracking', action='store_false', dest='tracking', help='Disable speaker tracking')
    
    parser.add_argument('--enable-zoom', action='store_true', default=True, help='Enable zoom punches')
    parser.add_argument('--no-zoom', action='store_false', dest='enable_zoom', help='Disable zoom punches')
    parser.add_argument('--enable-sfx', action='store_true', default=True, help='Enable sound effects')
    parser.add_argument('--no-sfx', action='store_false', dest='enable_sfx', help='Disable sound effects')
    parser.add_argument('--enable-progress', action='store_true', default=True, help='Enable progress bar')
    parser.add_argument('--no-progress', action='store_false', dest='enable_progress', help='Disable progress bar')
    
    args = parser.parse_args()
    
    if args.title_alias and not args.title:
        args.title = args.title_alias
    
    if args.duration > 90:
        print(f"Warning: Duration {args.duration}s is very long, recommended max: 90s")
    
    process_video(
        args.input,
        args.output,
        args.start,
        args.duration,
        args.ass_file,
        args.title,
        args.ai_mode,
        tracking=args.tracking,
        enable_zoom=args.enable_zoom,
        enable_sfx=args.enable_sfx,
        enable_progress=args.enable_progress
    )
    
    print("\\n=== Processing Complete (VIRAL EDITION)! ===")
