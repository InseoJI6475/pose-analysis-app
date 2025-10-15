# analyzer_module.py (v3.4 - 모든 기능 및 수정사항이 반영된 최종 완성본)

import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.signal import find_peaks # Cadence 계산을 위해 추가

# --- 1. 모델 및 라이브러리 초기화 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# 사용자 요청사항 반영: 인식률 상향 조정 (0.5 -> 0.3)
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.3)


# --- 2. 핵심 계산 함수들 ---
def calculate_angle(a, b, c):
    """세 점 사이의 각도를 계산하는 함수 (0-180도)"""
    if a is None or b is None or c is None: return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_angle_with_vertical(p1, p2):
    """두 점이 이루는 선과 수직선 사이의 각도를 계산하는 함수"""
    if p1 is None or p2 is None: return None
    p1, p2 = np.array(p1), np.array(p2)
    vector = p2 - p1
    vertical_vector = np.array([0, -1]) # Y축 위쪽 방향을 수직으로 정의
    dot_product = np.dot(vector, vertical_vector)
    norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical_vector)
    if norm_product == 0: return None
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    return np.degrees(angle_rad)


# --- 3. 전문가 진단 함수 (웹 호환성 강화) ---
def analyze_marching_pose(landmarks, visibility_threshold=0.5):
    """랜드마크 좌표를 바탕으로 모든 전문가 진단 패턴을 분석하는 함수"""
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
        if isinstance(value, (np.floating, np.integer, float, int)):
            sanitized_analysis[key] = float(value)
        elif value is None:
             sanitized_analysis[key] = 0.0
        else:
            sanitized_analysis[key] = value
            
    return sanitized_analysis

# --- 4. 위험도 점수 계산 함수 ---
def calculate_load_score(analysis, rucksack_weight_kg=20):
    score_mapping = {"양호": 0, "주의": 1, "위험": 2}
    weights = {"back": 0.5, "knee": 0.3, "neck": 0.2}
    base_score = sum(score_mapping.get(analysis.get(f"{part}_diagnosis"), 0) * weights[part] for part in weights)
    weight_factor = 1.5 if rucksack_weight_kg >= 25 else (1.2 if rucksack_weight_kg >= 15 else 1.0)
    return int(min(100, (base_score / 2.0) * 100 * weight_factor * 0.7))

# --- 5. 웹 서버가 호출할 메인 함수 ---
def run_analysis(filepath):
    try:
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in ['.mp4', '.mov', '.avi', '.mkv']:
            return analyze_video(filepath)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return analyze_image(filepath)
        else:
            return None, None, None
    except Exception as e:
        print(f"[ERROR in run_analysis]: {e}")
        return None, None, None

# --- 6. 이미지 분석 함수 ---
def analyze_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None: return None, None, None
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            pose_analysis = analyze_marching_pose(results.pose_landmarks.landmark)
            if not pose_analysis: return None, None, None
            load_score = calculate_load_score(pose_analysis)
            pose_analysis['load_score'] = float(load_score)
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            score_color = (0, 0, 255) if load_score > 70 else ((0, 255, 255) if load_score > 40 else (0, 255, 0))
            cv2.putText(annotated_image, f"Load Score: {int(load_score)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
            result_filename = 'result_' + os.path.basename(image_path)
            result_filepath = os.path.join('static', result_filename)
            cv2.imwrite(result_filepath, annotated_image)
            return pose_analysis, result_filepath, 'image'
        else:
            return None, None, None
    except Exception as e:
        print(f"[ERROR in analyze_image]: {e}")
        return None, None, None

# --- 7. 동영상 분석 함수 (Cadence 기능 포함) ---
def analyze_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        all_results, frames = [], []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        ankle_y_positions = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
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
        
        if not all_results: return None, None, None
        
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
        if ankle_y_positions and fps > 0:
            peaks, _ = find_peaks(np.array(ankle_y_positions) * -1, height=np.mean(np.array(ankle_y_positions) * -1), distance=fps*0.3)
            num_steps = len(peaks)
            video_duration_sec = len(frames) / fps
            if video_duration_sec > 0:
                cadence = (num_steps / video_duration_sec) * 60
        stats_result['cadence'] = float(cadence)
        
        worst_frame_analysis = max(all_results, key=lambda x: x.get('load_score', 0)) if all_results else {}
        for part in ['back', 'knee', 'neck']:
            if f'{part}_pattern' in worst_frame_analysis:
                stats_result[f'{part}_pattern'] = worst_frame_analysis[f'{part}_pattern']
        
        worst_frame_index = np.argmax(scores) if scores else 0
        if not frames: return None, None, None
        worst_frame = frames[worst_frame_index]
        
        results_for_worst_frame = pose.process(cv2.cvtColor(worst_frame, cv2.COLOR_BGR2RGB))
        if results_for_worst_frame.pose_landmarks:
            mp_drawing.draw_landmarks(worst_frame, results_for_worst_frame.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            max_score = stats_result['load_score']['max']
            score_color = (0, 0, 255) if max_score > 70 else ((0, 255, 255) if max_score > 40 else (0, 255, 0))
            cv2.putText(worst_frame, f"Max Score: {int(max_score)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        
        result_filename = 'result_' + os.path.basename(video_path) + '.jpg'
        result_filepath = os.path.join('static', result_filename)
        cv2.imwrite(result_filepath, worst_frame)
        
        sanitized_stats = {}
        for key, stats in stats_result.items():
            if isinstance(stats, dict):
                sanitized_stats[key] = {k: float(v) if not np.isnan(v) else 0 for k, v in stats.items()}
            else:
                sanitized_stats[key] = stats
                
        return sanitized_stats, result_filepath, 'video'
        
    except Exception as e:
        print(f"[ERROR in analyze_video]: {e}")
        return None, None, None