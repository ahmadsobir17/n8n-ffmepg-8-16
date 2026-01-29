#!/usr/bin/env python3
"""
Face Detection Auto-Crop for Podcast Videos with Whisper Transcription
Uses MediaPipe to detect faces and Whisper to generate word-timed subtitles.
Enhanced with dynamic face tracking and robust error handling.
"""

from pathlib import Path
import os
import tempfile
import sys
import argparse
import subprocess
import json

# Suppress MediaPipe/TF/OpenCV logging and force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_GPU_DISABLED'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ["GLOG_minloglevel"] = "2"

try:
    import cv2
    import mediapipe as mp
    import whisper
    import gc
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install opencv-python-headless mediapipe openai-whisper")
    sys.exit(1)


class FaceDetector:
    """MediaPipe-based face detector for video frames."""
    
    def __init__(self, min_confidence=0.4):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Better for wide shots/podcast layouts
            min_detection_confidence=min_confidence
        )
    
    def close(self):
        """Clean up MediaPipe detector."""
        if hasattr(self, 'detector'):
            self.detector.close()
    
    def detect_faces(self, frame):
        if frame is None:
            return []
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = self.detector.process(rgb_frame)
        except Exception as e:
            print(f"  Debug: MediaPipe error during process: {e}")
            return []
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                fx = int(bbox.xmin * w)
                fy = int(bbox.ymin * h)
                fw = int(bbox.width * w)
                fh = int(bbox.height * h)
                # Return relative coordinates for easier processing
                faces.append({
                    'x': fx, 'y': fy, 'w': fw, 'h': fh,
                    'rel_x': bbox.xmin,
                    'rel_y': bbox.ymin,
                    'rel_w': bbox.width,
                    'rel_h': bbox.height,
                    'center_x': bbox.xmin + bbox.width / 2,
                    'center_y': bbox.ymin + bbox.height / 2
                })
        
        return faces


def extract_segment_h264(video_path, start_time, duration):
    """Extract a video segment and re-encode to H.264 for better OpenCV compatibility."""
    temp_segment = tempfile.NamedTemporaryFile(suffix='_segment.mp4', delete=False)
    temp_segment.close()
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', video_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '23',
        '-an',
        temp_segment.name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: Failed to extract H.264 segment: {result.stderr[:200]}")
        return None
    
    return temp_segment.name


def get_video_info(video_path):
    """Get video dimensions and duration using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None, None, None
    
    try:
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        fmt = data.get('format', {})
        width = stream.get('width', 1920)
        height = stream.get('height', 1080)
        duration = float(stream.get('duration') or fmt.get('duration', 0))
        return width, height, duration
    except:
        return None, None, None


def sample_frames_from_segment(segment_path, sample_interval=1.0):
    """Sample frames from a pre-extracted H.264 segment."""
    cap = cv2.VideoCapture(segment_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames_count = max(1, int(fps * sample_interval))
    
    frames = []
    current_frame = 0
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = current_frame / fps
        frames.append((timestamp, frame))
        current_frame += sample_frames_count
    
    cap.release()
    return frames


def analyze_video(video_path, start_time, duration):
    """Analyze video to determine face positions and camera angles with tracking data."""
    detector = FaceDetector()
    temp_segment = None
    
    try:
        # Get video info using ffprobe (more reliable)
        video_width, video_height, video_duration = get_video_info(video_path)
        
        if video_width is None:
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        
        print(f"  Video stats: {video_width}x{video_height}, Duration: {video_duration:.1f}s")
        
        if start_time >= video_duration:
            print(f"  Warning: Start time ({start_time}s) is beyond video duration ({video_duration:.1f}s)!")
            start_time = max(0, video_duration - duration - 1)
            print(f"  Adjusted start time to: {start_time:.1f}s")
        
        # Calculate aspect ratio hint
        aspect_ratio = video_width / video_height if video_height > 0 else 1.0
        is_ultrawide = aspect_ratio > 2.0  # e.g., 21:9 is ~2.33
        print(f"  Aspect ratio: {aspect_ratio:.2f} ({'Ultrawide' if is_ultrawide else 'Standard'})")

        # Extract H.264 segment for reliable frame reading
        print("  Extracting H.264 segment for face detection...")
        temp_segment = extract_segment_h264(video_path, start_time, duration)
        
        if not temp_segment or not os.path.exists(temp_segment):
            print("  Error: Failed to extract video segment")
            return create_fallback_analysis(video_width, video_height, duration)
        
        frames = sample_frames_from_segment(temp_segment, sample_interval=1.0)
        print(f"  Sampled {len(frames)} frames")
        
        if not frames:
            return create_fallback_analysis(video_width, video_height, duration)
        
        # Analyze each frame for face positions
        frame_data = []
        
        for i, (timestamp, frame) in enumerate(frames):
            if frame is None:
                continue
                
            faces = detector.detect_faces(frame)
            
            # Classify frame type based on face positions
            left_faces = [f for f in faces if f['center_x'] < 0.5]
            right_faces = [f for f in faces if f['center_x'] >= 0.5]
            
            is_wide = len(faces) >= 2 or (len(left_faces) > 0 and len(right_faces) > 0)
            
            # Get primary speaker position (largest face or rightmost in wide shot)
            if faces:
                # Sort by face size (largest first)
                faces_by_size = sorted(faces, key=lambda f: f['rel_w'] * f['rel_h'], reverse=True)
                primary_face = faces_by_size[0]
            else:
                primary_face = {'center_x': 0.5, 'center_y': 0.5, 'rel_w': 0.2, 'rel_h': 0.3}
            
            frame_data.append({
                'timestamp': timestamp,
                'is_wide': is_wide,
                'face_count': len(faces),
                'faces': faces,
                'primary_x': primary_face['center_x'],
                'primary_y': primary_face['center_y'],
                'left_faces': left_faces,
                'right_faces': right_faces
            })
        
        if not frame_data:
            return create_fallback_analysis(video_width, video_height, duration)
        
        # Segment analysis: group frames into scenes with consistent modes
        segments = []
        if frame_data:
            # Determine mode for each frame
            raw_modes = []
            for f in frame_data:
                # Wide if:
                # 1. 2+ faces found
                # 2. At least one on left AND one on right
                # 3. Ultrawide video and NO faces detected (safe default for podcasts)
                is_wide_evidence = (f['face_count'] >= 2) or (len(f['left_faces']) > 0 and len(f['right_faces']) > 0)
                
                # For ultrawide videos (podcasts), use smart detection:
                # - Wide shot: 2+ faces, or 1 small/off-center face
                # - Close-up shot: 1 LARGE centered face (face takes up significant portion of frame)
                if is_ultrawide:
                    if f['face_count'] == 0:
                        mode = 'wide'  # No faces = assume wide intro/set shot
                    elif f['face_count'] == 1:
                        # Check if this is a TRUE close-up (large face, centered)
                        face = f['faces'][0] if f['faces'] else None
                        if face:
                            face_size = face['rel_w'] * face['rel_h']
                            # Wide-shot faces: ~0.01, Close-up faces: ~0.05
                            # Threshold 0.025 = midpoint to cleanly separate them
                            is_large_face = face_size > 0.025  # Face > 2.5% of frame = true close-up
                            is_centered = abs(f['primary_x'] - 0.5) < 0.25  # Within 0.25-0.75 range
                            mode = 'closeup' if (is_large_face and is_centered) else 'wide'
                        else:
                            mode = 'wide'
                    else:
                        mode = 'wide'  # 2+ faces = always wide
                else:
                    # For standard aspect ratios, use face detection
                    mode = 'wide' if is_wide_evidence else 'closeup'
                
                raw_modes.append(mode)
            
            # Simple debouncing/smoothing of modes (min 2 seconds per segment)
            # This prevents flickering between modes
            smoothed_modes = []
            for i in range(len(raw_modes)):
                window = raw_modes[max(0, i-2):min(len(raw_modes), i+3)]
                # Majority vote
                smoothed_modes.append(max(set(window), key=window.count))
            
            current_mode = smoothed_modes[0]
            segment_start = 0
            for i in range(1, len(smoothed_modes)):
                if smoothed_modes[i] != current_mode:
                    segments.append({
                        'start': segment_start,
                        'end': frame_data[i]['timestamp'],
                        'mode': current_mode
                    })
                    segment_start = frame_data[i]['timestamp']
                    current_mode = smoothed_modes[i]
            
            segments.append({
                'start': segment_start,
                'end': duration,
                'mode': current_mode
            })

        # Calculate primary face for each segment to avoid jitter
        for seg in segments:
            seg_frames = [f for f in frame_data if seg['start'] <= f['timestamp'] <= seg['end']]
            if seg_frames:
                seg['avg_x'] = sum(f['primary_x'] for f in seg_frames) / len(seg_frames)
                seg['avg_y'] = sum(f['primary_y'] for f in seg_frames) / len(seg_frames)
            else:
                seg['avg_x'] = 0.5
                seg['avg_y'] = 0.5

        return {
            'video_width': video_width,
            'video_height': video_height,
            'video_duration': video_duration,
            'segments': segments,
            'avg_face_count': sum(f['face_count'] for f in frame_data) / len(frame_data) if frame_data else 0,
            'adjusted_start_time': start_time,
            'aspect_ratio': aspect_ratio,
            'is_ultrawide': is_ultrawide
        }
        
    finally:
        detector.close()
        if temp_segment and os.path.exists(temp_segment):
            try:
                os.unlink(temp_segment)
            except:
                pass


def create_fallback_analysis(width, height, duration):
    """Create fallback analysis when face detection fails."""
    return {
        'video_width': width,
        'video_height': height,
        'video_duration': duration,
        'segments': [{'start': 0, 'end': duration, 'mode': 'closeup', 'avg_x': 0.5, 'avg_y': 0.5}],
        'avg_face_count': 0,
        'adjusted_start_time': 0,
        'aspect_ratio': 1.0,
        'is_ultrawide': False
    }


def smooth_tracking_data(frame_data, window=3):
    """Apply smoothing to face tracking data to reduce jitter."""
    if len(frame_data) < window:
        return frame_data
    
    smoothed = []
    for i in range(len(frame_data)):
        start = max(0, i - window // 2)
        end = min(len(frame_data), i + window // 2 + 1)
        window_frames = frame_data[start:end]
        
        # Weighted average (current frame has more weight)
        total_weight = 0
        avg_x = 0
        avg_y = 0
        
        for j, f in enumerate(window_frames):
            weight = 2 if j == len(window_frames) // 2 else 1
            avg_x += f['primary_x'] * weight
            avg_y += f['primary_y'] * weight
            total_weight += weight
        
        smoothed_frame = frame_data[i].copy()
        smoothed_frame['primary_x'] = avg_x / total_weight
        smoothed_frame['primary_y'] = avg_y / total_weight
        smoothed.append(smoothed_frame)
    
    return smoothed


def generate_crop_filter(analysis, duration, force_mode='auto'):
    """Generate dynamic FFmpeg filter that can switch between modes."""
    w = analysis['video_width']
    h = analysis['video_height']
    segments = analysis.get('segments', [])
    
    out_w = 1080
    out_h = 1920
    panel_h = out_h // 2
    aspect_9_16 = 9 / 16
    aspect_panel = out_w / panel_h
    
    if force_mode != 'auto':
        for seg in segments:
            seg['mode'] = force_mode

    filter_parts = []
    seg_names = []
    
    for i, seg in enumerate(segments):
        start = seg['start']
        end = seg['end']
        mode = seg['mode']
        avg_x = seg.get('avg_x', 0.5)
        
        seg_id = f"v{i}"
        trim = f"trim=start={start}:end={end},setpts=PTS-STARTPTS"
        
        if mode == 'wide':
            # Wide mode calculations (Host top, Speaker bottom)
            crop_w = min(w // 2 - 20, int(h * aspect_panel))
            crop_h = min(h, int(crop_w / aspect_panel))
            crop_w = max(100, min(crop_w, w // 2))
            crop_h = max(100, min(crop_h, h))
            left_x = max(0, int(w * 0.25) - crop_w // 2)
            left_x = min(left_x, w // 2 - crop_w)
            right_x = max(w // 2, int(w * 0.75) - crop_w // 2)
            right_x = min(right_x, w - crop_w)
            crop_y = max(0, (h - crop_h) // 2)
            
            part = (
                f"[0:v]{trim},split=2[t{i}][b{i}];"
                f"[t{i}]crop={crop_w}:{crop_h}:{left_x}:{crop_y},scale={out_w}:{panel_h},setsar=1,fps=24,format=yuv420p[top{i}];"
                f"[b{i}]crop={crop_w}:{crop_h}:{right_x}:{crop_y},scale={out_w}:{panel_h},setsar=1,fps=24,format=yuv420p[bot{i}];"
                f"[top{i}][bot{i}]vstack=inputs=2[{seg_id}]"
            )
        else:
            # Close-up mode calculations
            crop_h = h
            crop_w = int(h * aspect_9_16)
            if crop_w > w:
                crop_w = w
                crop_h = int(w / aspect_9_16)
                crop_h = min(crop_h, h)
            crop_x = int(avg_x * w) - crop_w // 2
            crop_x = max(0, min(crop_x, w - crop_w))
            crop_y = 0
            
            part = f"[0:v]{trim},crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h},setsar=1,fps=24,format=yuv420p[{seg_id}]"
            
        filter_parts.append(part)
        seg_names.append(f"[{seg_id}]")
    
    # Concat all segments with smooth crossfade transitions
    if len(segments) > 1:
        # Use slideup for professional-looking transitions (0.5s)
        fade_duration = 0.5
        
        # Build xfade chain progressively
        xfade_parts = []
        prev_output = seg_names[0]  # Start with [v0]
        cumulative_duration = segments[0]['end'] - segments[0]['start']
        
        for i in range(1, len(segments)):
            next_seg = seg_names[i]
            
            # Output label: intermediate [x0], [x1], etc. or final [stacked]
            if i < len(segments) - 1:
                output = f"[x{i-1}]"
            else:
                output = "[stacked]"
            
            # Offset = cumulative duration so far minus fade overlap
            offset = max(0, cumulative_duration - fade_duration)
            
            xfade = f"{prev_output}{next_seg}xfade=transition=slideup:duration={fade_duration}:offset={offset:.3f}{output}"
            xfade_parts.append(xfade)
            
            # Update cumulative duration (add next segment, minus fade overlap)
            seg_duration = segments[i]['end'] - segments[i]['start']
            cumulative_duration = offset + seg_duration
            prev_output = output
        
        filter_str = ";".join(filter_parts) + ";" + ";".join(xfade_parts)
    else:
        # Just rename the single segment to [stacked]
        last_seg_name = seg_names[0]
        filter_str = filter_parts[0].replace(last_seg_name, "[stacked]")
    
    return filter_str, len(segments) > 1


def transcribe_audio(video_path, start_time, duration, model_name='tiny'):
    """Transcribe audio from video segment using Whisper."""
    print(f"  Transcribing audio with Whisper ({model_name})...")
    
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            temp_audio.name
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: Audio extraction failed")
            return []
        
        model = whisper.load_model(model_name)
        result = model.transcribe(
            temp_audio.name,
            word_timestamps=True,
            language=None
        )
        
        words = []
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                words.append({
                    'word': word_info['word'].strip(),
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        print(f"  Transcribed {len(words)} words")
        
        del model
        gc.collect()
        
        return words
        
    except Exception as e:
        print(f"  Whisper error: {e}")
        return []
    finally:
        if os.path.exists(temp_audio.name):
            os.unlink(temp_audio.name)


def format_ass_time(seconds):
    """Format seconds to ASS time format: H:MM:SS.CC"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def generate_ass_subtitle(words, is_wide, duration):
    """Generate ASS subtitle with word-by-word timing."""
    # Style based on mode
    if is_wide:
        margin_v = 880  # Between panels
        font_size = 48
    else:
        margin_v = 120  # Bottom of screen
        font_size = 54
    
    style_name = "MainStyle"
    
    ass_content = f"""[Script Info]
Title: Auto-Generated Subtitle
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},Arial Black,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&HC0000000,-1,0,0,0,100,100,2,0,1,4,2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    if not words:
        # Empty subtitle if no words
        ass_content += f"Dialogue: 0,0:00:00.00,0:00:01.00,{style_name},,0,0,0,,\n"
        return ass_content
    
    # Group words into phrases (4-5 words)
    phrases = []
    current_phrase = []
    current_start = None
    
    for word_info in words:
        if current_start is None:
            current_start = word_info['start']
        
        current_phrase.append(word_info['word'])
        
        if len(current_phrase) >= 4 or word_info['word'].endswith(('.', '?', '!', ',')):
            phrases.append({
                'text': ' '.join(current_phrase),
                'start': current_start,
                'end': word_info['end']
            })
            current_phrase = []
            current_start = None
    
    if current_phrase:
        phrases.append({
            'text': ' '.join(current_phrase),
            'start': current_start,
            'end': words[-1]['end']
        })
    
    for phrase in phrases:
        start_time = format_ass_time(phrase['start'])
        end_time = format_ass_time(phrase['end'])
        text = phrase['text'].replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
        animated_text = r"{\fad(100,100)}" + text
        ass_content += f"Dialogue: 0,{start_time},{end_time},{style_name},,0,0,0,,{animated_text}\n"
    
    return ass_content


def run_ffmpeg(cmd, description="FFmpeg"):
    """Run FFmpeg command with proper error handling."""
    print(f"  Running {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  {description} failed (code {result.returncode})")
        # Print last 500 chars of stderr for debugging
        if result.stderr:
            print(f"  Error: {result.stderr[-500:]}")
        return False
    return True


def process_video(input_file, output_file, start_time, duration, ass_file=None, use_whisper=True, mode='auto'):
    """Process video with face detection, smart cropping, and optional subtitles."""
    print(f"Processing video: {input_file}")
    print(f"  Start: {start_time}s, Duration: {duration}s, Mode: {mode}")
    
    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Analyze video for face positions
    analysis = analyze_video(input_file, start_time, duration)
    
    # Generate crop filter
    crop_filter, is_dynamic = generate_crop_filter(analysis, duration, force_mode=mode)
    
    # Decide if any wide layout exists for subtitle positioning
    has_wide = any(seg['mode'] == 'wide' for seg in analysis.get('segments', []))
    print(f"  Layout: {'Dynamic' if is_dynamic else 'Static'} ({'Contains Wide' if has_wide else 'All Closeup'})")
    
    # Use adjusted start time from analysis (may be different if original was out of bounds)
    actual_start_time = analysis.get('adjusted_start_time', start_time)
    if actual_start_time != start_time:
        print(f"  Using adjusted start time: {actual_start_time:.1f}s (original: {start_time}s)")
    
    # Transcribe and generate subtitles
    ass_content = None
    if use_whisper and ass_file:
        try:
            words = transcribe_audio(input_file, actual_start_time, duration, model_name='tiny')
            ass_content = generate_ass_subtitle(words, has_wide, duration)
            with open(ass_file, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            print(f"  Subtitle saved to: {ass_file}")
        except Exception as e:
            print(f"  Subtitle generation failed: {e}")
            ass_file = None
    
    # Build FFmpeg command
    if ass_file and os.path.exists(ass_file):
        full_filter = f"{crop_filter};[stacked]ass={ass_file}[outv]"
        output_stream = "[outv]"
    else:
        full_filter = crop_filter
        output_stream = "[stacked]"
    
    print(f"  Filter: {crop_filter[:100]}...")
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(actual_start_time),
        '-t', str(duration),
        '-i', input_file,
        '-filter_complex', full_filter,
        '-map', output_stream,
        '-map', '0:a?',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        output_file
    ]
    
    success = run_ffmpeg(cmd, "FFmpeg (with subtitles)" if ass_file else "FFmpeg")
    
    # If failed, try without subtitles
    if not success and ass_file:
        print("  Retrying without subtitles...")
        cmd_simple = [
            'ffmpeg', '-y',
            '-ss', str(actual_start_time),
            '-t', str(duration),
            '-i', input_file,
            '-filter_complex', crop_filter,
            '-map', '[stacked]',
            '-map', '0:a?',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_file
        ]
        success = run_ffmpeg(cmd_simple, "FFmpeg (no subtitles)")
    
    # If still failed, try simple center crop as last resort
    if not success or not os.path.exists(output_file) or os.path.getsize(output_file) < 10000:
        print("  Using fallback simple crop...")
        w = analysis['video_width']
        h = analysis['video_height']
        crop_w = min(w, int(h * 9 / 16))
        crop_x = (w - crop_w) // 2
        
        cmd_fallback = [
            'ffmpeg', '-y',
            '-ss', str(actual_start_time),
            '-t', str(duration),
            '-i', input_file,
            '-vf', f'crop={crop_w}:{h}:{crop_x}:0,scale=1080:1920',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',
            output_file
        ]
        success = run_ffmpeg(cmd_fallback, "FFmpeg (fallback crop)")
    
    # Final validation
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        if file_size < 10000:
            print(f"  ERROR: Output file is too small ({file_size} bytes)")
            raise RuntimeError(f"FFmpeg produced invalid output ({file_size} bytes)")
        print(f"  Success! Output: {output_file} ({file_size / 1024:.1f} KB)")
    else:
        raise RuntimeError("FFmpeg failed to produce output file")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Face detection auto-crop with Whisper subtitles')
    parser.add_argument('--input', '-i', required=True, help='Input video file')
    parser.add_argument('--output', '-o', required=True, help='Output video file')
    parser.add_argument('--start', '-s', type=float, default=0, help='Start time in seconds')
    parser.add_argument('--duration', '-d', type=float, default=60, help='Duration in seconds')
    parser.add_argument('--ass-file', help='Path to save ASS subtitle file')
    parser.add_argument('--no-whisper', action='store_true', help='Disable Whisper transcription')
    parser.add_argument('--mode', choices=['auto', 'wide', 'closeup'], default='auto', help='Crop mode (default: auto)')
    
    args = parser.parse_args()
    
    try:
        process_video(
            args.input,
            args.output,
            args.start,
            args.duration,
            args.ass_file,
            use_whisper=not args.no_whisper,
            mode=args.mode
        )
        print("Processing complete!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
