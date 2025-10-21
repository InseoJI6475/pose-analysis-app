# analyzer_module.py (Cloudinary 연동 최종 완성본)

import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import find_peaks
import urllib.request # URL 이미지 처리를 위해 추가
import requests # 동영상 다운로드를 위해 추가

# --- 1. 모델 및 라이브러리 초기화 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.3)

# 임시 파일 저장 경로 (app.py와 동일한 위치 사용)
TEMP_FOLDER = '/tmp'
if not os.path.exists(TEMP_FOLDER):
    try:
        os.makedirs(TEMP_FOLDER)
    except OSError as e:
        print(f"Analyzer: Could not create temp folder {TEMP_FOLDER}: {e}")
        TEMP_FOLDER = 'static/uploads' # 생성 실패 시 이전 방식 사용 (로컬 테스트용)
        if not os.path.exists(TEMP_FOLDER):
             os.makedirs(TEMP_FOLDER)


# --- 2. 핵심 계산 함수들 (calculate_angle, calculate_angle_with_vertical) ---
def calculate_angle(a, b, c):
    if a is None or b is None or c is None: return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_angle_with_vertical(p1, p2):
    if p1 is None or p2 is None: return None
    p1, p2 = np.array(p1), np.array(p2)
    vector = p2 - p1
    vertical_vector = np.array([0, -1])
    dot_product = np.dot(vector, vertical_vector)
    norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical_vector)
    if norm_product == 0: return None
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle_rad)


# --- 3. 전문가 진단 함수 (analyze_marching_pose) ---
def analyze_marching_pose(landmarks, visibility_threshold=0.5):
    analysis = {}
    try:
        side = "LEFT" if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility else "RIGHT"
        def get_coords(name, side_override=None):
            side_to_use = side_override or side
            lm = landmarks[mp_pose.PoseLandmark[f"{side_to_use}_{name}"].value]
            return [lm.x, lm.y] if lm.visibility > visibility_threshold else None
        shoulder, hip, knee, ankle, ear = (get_coords(name) for name in ["SHOULDER", "HIP", "KNEE", "ANKLE", "EAR"])
        left_hip, right_hip = get_coords("HIP", "LEFT"), get_coords("HIP", "RIGHT")
        analysis['back_angle'] = calculate_angle_with_vertical(hip, shoulder)
        analysis['knee_angle'] = calculate_angle(hip, knee, ankle)
        analysis['neck_angle'] = calculate_angle(ear, shoulder, hip)
        if analysis.get('back_angle') is not None:
            lean = analysis['back_angle']
            if lean < 15: analysis['back_diagnosis'] = "양호"
            elif 15 <= lean < 25: analysis.update({'back_diagnosis': "주의", 'back_pattern': "상체 과다 숙임"})
            else:
                analysis['back_diagnosis'] = "위험"
                if shoulder and left_hip and right_hip and abs(shoulder[0] - (left_hip[0] + right_hip[0]) / 2) > 0.05: analysis['back_pattern'] = "골반 측방 붕괴"
                else: analysis['back_pattern'] = "흉·요추 후만 변형"
        if analysis.get('knee_angle') is not None:
            angle = analysis['knee_angle']
            if 185 > angle > 175: analysis.update({'knee_diagnosis': "위험", 'knee_pattern': "무릎 과신전 (Genu Recurvatum)"})
            elif 175 >= angle > 160: analysis.update({'knee_diagnosis': "주의", 'knee_pattern': "충격 흡수 부전"})
            elif 160 >= angle > 140: analysis['knee_diagnosis'] = "양호"
            elif 140 >= angle > 120: analysis.update({'knee_diagnosis': "주의", 'knee_pattern': "과도한 무릎 굽힘"})
            else: analysis.update({'knee_diagnosis': "위험", 'knee_pattern': "대퇴사두근 우세 패턴"})
        if analysis.get('neck_angle') is not None:
            angle = analysis['neck_angle']
            if angle > 150: analysis['neck_diagnosis'] = "양호"
            elif 150 >= angle > 135: analysis.update({'neck_diagnosis': "주의", 'neck_pattern': "전방 머리 자세"})
            else:
                analysis['neck_diagnosis'] = "위험"
                thoracic_angle = calculate_angle(ankle, hip, shoulder) if shoulder and hip and ankle else 180
                if thoracic_angle < 160: analysis['neck_pattern'] = "굽은 등 보상 패턴"
                else: analysis['neck_pattern'] = "거북목 증후군"
    except Exception as e:
        print(f"[ERROR in analyze_marching_pose]: {e}")
        return {}
    sanitized_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, (np.floating, np.integer, float, int)): sanitized_analysis[key] = float(value)
        elif value is None: sanitized_analysis[key] = 0.0
        else: sanitized_analysis[key] = value
    return sanitized_analysis

# --- 4. 위험도 점수 계산 함수 (calculate_load_score) ---
def calculate_load_score(analysis, rucksack_weight_kg=20):
    score_mapping = {"양호": 0, "주의": 1, "위험": 2}
    weights = {"back": 0.5, "knee": 0.3, "neck": 0.2}
    base_score = sum(score_mapping.get(analysis.get(f"{part}_diagnosis"), 0) * weights[part] for part in weights)
    weight_factor = 1.5 if rucksack_weight_kg >= 25 else (1.2 if rucksack_weight_kg >= 15 else 1.0)
    return int(min(100, (base_score / 2.0) * 100 * weight_factor * 0.7))


# --- 5. 웹 서버가 호출할 메인 함수들 ---

# 기존 run_analysis 함수 (app.py에서 동영상 다운로드 후 호출됨)
def run_analysis(filepath):
    """로컬 파일 경로를 받아 분석을 실행하는 함수 (주로 동영상 처리용)"""
    try:
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in ['.mp4', '.mov', '.avi', '.mkv']:
            return analyze_video(filepath) # 로컬 동영상 파일 분석
        # 이미지의 경우 run_analysis_from_url 사용 권장
        elif file_ext in ['.jpg', '.jpeg', '.png']:
             print("Warning: run_analysis called for image, use run_analysis_from_url instead for URL input.")
             image = cv2.imread(filepath)
             if image is not None:
                 return analyze_image_data(image, filename)
             else:
                 return None, None, None
        else:
            return None, None, None
    except Exception as e:
        print(f"[ERROR in run_analysis for {filepath}]: {e}")
        return None, None, None

# URL을 입력받는 새로운 메인 함수 (주로 이미지 처리용)
def run_analysis_from_url(media_url):
    """미디어 URL을 받아 분석을 실행하는 함수 (주로 이미지 처리용)"""
    try:
        print(f"Analyzing media from URL: {media_url}")
        # --- URL에서 이미지 데이터를 메모리로 직접 다운로드 ---
        req = urllib.request.urlopen(media_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR) # 컬러 이미지로 로드

        if image is None:
            print("Failed to decode image from URL.")
            return None, None, None

        # 파일 이름은 URL에서 추출 (결과 저장용)
        filename = os.path.basename(media_url.split('?')[0]) # URL 파라미터 제거

        # 이미지는 바로 analyze_image_data 호출
        return analyze_image_data(image, filename)

    except Exception as e:
        print(f"[ERROR in run_analysis_from_url for {media_url}]: {e}")
        return None, None, None


# --- 6. 이미지 분석 함수 (이미지 데이터를 직접 받음) ---
def analyze_image_data(image, filename):
    """이미지 데이터(numpy array)와 파일 이름을 받아 분석하는 함수"""
    try:
        print(f"Analyzing image data for: {filename}")
        # 클라이언트에서 리사이즈 했더라도, 안전을 위해 서버에서도 체크/리사이즈 (선택적)
        h, w, _ = image.shape
        max_dim = 1080
        if h > max_dim or w > max_dim:
            print(f"Resizing image from {w}x{h} to max {max_dim}px")
            if h > w: new_h, new_w = max_dim, int(w * max_dim / h)
            else: new_w, new_h = max_dim, int(h * max_dim / w)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            pose_analysis = analyze_marching_pose(results.pose_landmarks.landmark)
            if not pose_analysis:
                print("Failed to analyze pose landmarks.")
                return None, None, None
            load_score = calculate_load_score(pose_analysis)
            pose_analysis['load_score'] = float(load_score)

            # --- 결과 이미지 생성 ---
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            score_color = (0, 0, 255) if load_score > 70 else ((0, 255, 255) if load_score > 40 else (0, 255, 0))
            cv2.putText(annotated_image, f"Load Score: {int(load_score)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

            # --- 결과 이미지를 로컬 임시 파일로 저장 (Cloudinary 업로드용) ---
            result_filename = f"result_{os.path.splitext(filename)[0]}.jpg" # 항상 jpg로 저장
            result_filepath = os.path.join(TEMP_FOLDER, result_filename)
            print(f"Saving temporary result image to: {result_filepath}")
            success = cv2.imwrite(result_filepath, annotated_image)
            if not success:
                print(f"Failed to write result image to {result_filepath}")
                return None, None, None

            # 반환값: 분석 데이터, 결과 이미지 *로컬 경로*, 타입
            return pose_analysis, result_filepath, 'image'
        else:
            print("No pose landmarks detected in the image.")
            return None, None, None
    except Exception as e:
        print(f"[ERROR in analyze_image_data for {filename}]: {e}")
        return None, None, None


# --- 7. 동영상 분석 함수 (로컬 임시 파일을 받음) ---
def analyze_video(video_path):
    """로컬 비디오 파일 경로를 받아 분석하는 함수"""
    try:
        print(f"Analyzing video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"Error opening video file: {video_path}")
             return None, None, None

        all_results = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        ankle_y_positions = []
        frame_count = 0
        max_frames_to_process = 300 # 메모리 제한을 위해 최대 프레임 수 제한 (약 10초 @ 30fps)

        while cap.isOpened() and frame_count < max_frames_to_process:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # 동영상 프레임 리사이즈 (메모리 절약)
            h, w, _ = frame.shape
            max_dim = 720
            if h > max_dim or w > max_dim:
                if h > w: new_h, new_w = max_dim, int(w * max_dim / h)
                else: new_w, new_h = max_dim, int(h * max_dim / w)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                side = "LEFT" if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > 0.5 else "RIGHT"
                ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark[f"{side}_ANKLE"].value].y
                ankle_y_positions.append(ankle_y)
                pose_analysis = analyze_marching_pose(results.pose_landmarks.landmark)
                if pose_analysis:
                    load_score = calculate_load_score(pose_analysis)
                    pose_analysis['load_score'] = float(load_score)
                    all_results.append(pose_analysis)
        cap.release()
        print(f"Processed {frame_count} frames.")

        if not all_results:
             print("No pose detected in any video frame.")
             return None, None, None

        scores = [r['load_score'] for r in all_results if 'load_score' in r]
        back_angles = [r['back_angle'] for r in all_results if r and r.get('back_angle')]
        knee_angles = [r['knee_angle'] for r in all_results if r and r.get('knee_angle')]
        neck_angles = [r['neck_angle'] for r in all_results if r and r.get('neck_angle')]

        stats_result = {
            'load_score': {'avg': np.mean(scores) if scores else 0, 'max': np.max(scores) if scores else 0, 'min': np.min(scores) if scores else 0, 'std': np.std(scores) if scores else 0},
            'back_angle': {'avg': np.mean(back_angles) if back_angles else 0, 'max': np.max(back_angles) if back_angles else 0, 'min': np.min(back_angles) if back_angles else 0, 'std': np.std(back_angles) if back_angles else 0},
            'knee_angle': {'avg': np.mean(knee_angles) if knee_angles else 0, 'max': np.max(knee_angles) if knee_angles else 0, 'min': np.min(knee_angles) if knee_angles else 0, 'std': np.std(knee_angles) if knee_angles else 0},
            'neck_angle': {'avg': np.mean(neck_angles) if neck_angles else 0, 'max': np.max(neck_angles) if neck_angles else 0, 'min': np.min(neck_angles) if neck_angles else 0, 'std': np.std(neck_angles) if neck_angles else 0}
        }

        cadence = 0
        if ankle_y_positions and fps > 0 and fps < 100: # 비정상적인 fps 값 방지
            try:
                peaks, _ = find_peaks(np.array(ankle_y_positions) * -1, height=np.mean(np.array(ankle_y_positions) * -1), distance=fps*0.3)
                num_steps = len(peaks)
                video_duration_sec = frame_count / fps
                if video_duration_sec > 0:
                    cadence = (num_steps / video_duration_sec) * 60
                print(f"Cadence calculated: {cadence:.1f} steps/min")
            except Exception as peak_err:
                print(f"Could not calculate cadence: {peak_err}")
        stats_result['cadence'] = float(cadence)

        worst_frame_analysis = max(all_results, key=lambda x: x.get('load_score', 0)) if all_results else {}
        for part in ['back', 'knee', 'neck']:
            if f'{part}_pattern' in worst_frame_analysis:
                stats_result[f'{part}_pattern'] = worst_frame_analysis[f'{part}_pattern']

        # --- 최악의 프레임만 다시 로드하여 결과 이미지 생성 ---
        worst_frame_index_in_results = np.argmax(scores) if scores else 0
        # 실제 프레임 인덱스 찾기 (분석 성공한 프레임 기준)
        processed_frame_indices = [i for i, r in enumerate(all_results) if r] # 분석 성공한 프레임들의 원래 인덱스 (0부터 시작)
        worst_actual_frame_index = processed_frame_indices[worst_frame_index_in_results] if processed_frame_indices else 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None, None, None
        cap.set(cv2.CAP_PROP_POS_FRAMES, worst_actual_frame_index)
        ret, worst_frame = cap.read()
        cap.release()

        result_filepath = None
        if ret:
            h, w, _ = worst_frame.shape
            max_dim = 1080
            if h > max_dim or w > max_dim:
                if h > w: new_h, new_w = max_dim, int(w * max_dim / h)
                else: new_w, new_h = max_dim, int(h * max_dim / w)
                worst_frame = cv2.resize(worst_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            results_for_worst_frame = pose.process(cv2.cvtColor(worst_frame, cv2.COLOR_BGR2RGB))
            if results_for_worst_frame.pose_landmarks:
                mp_drawing.draw_landmarks(worst_frame, results_for_worst_frame.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                max_score = stats_result['load_score']['max']
                score_color = (0, 0, 255) if max_score > 70 else ((0, 255, 255) if max_score > 40 else (0, 255, 0))
                cv2.putText(worst_frame, f"Max Score: {int(max_score)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

            # 결과 이미지를 로컬 임시 파일로 저장
            result_filename = f"result_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
            result_filepath = os.path.join(TEMP_FOLDER, result_filename)
            print(f"Saving temporary result image for video to: {result_filepath}")
            success = cv2.imwrite(result_filepath, worst_frame)
            if not success:
                print(f"Failed to write result image for video to {result_filepath}")
                result_filepath = None # 저장 실패 시 None 반환
        else:
             print("Failed to retrieve the worst frame from video.")


        # --- 결과 데이터 정리 (웹 호환성) ---
        sanitized_stats = {}
        for key, stats in stats_result.items():
            if isinstance(stats, dict):
                sanitized_stats[key] = {k: float(v) if not np.isnan(v) else 0 for k, v in stats.items()}
            else:
                sanitized_stats[key] = stats

        return sanitized_stats, result_filepath, 'video'
    except Exception as e:
        print(f"[ERROR in analyze_video for {video_path}]: {e}")
        return None, None, None