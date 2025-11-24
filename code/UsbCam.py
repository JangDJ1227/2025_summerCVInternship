from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import time
import os
import json
from multiprocessing import Process, Queue
import queue
import torch
import gc

# 클래스 파일 로드
def load_classes(classes_file="letterDetect_new/classes.txt"):
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip().split(': ')[1] for line in f]
    return classes

# 한국 번호판 한글 클래스
def get_korean_plate_chars(classes):
    return ''.join(classes[10:72])



# 한국 지역명 리스트
region_names = ['서울', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '대구', '인천', '광주', '대전', '울산', '부산']

# 재시도 임계값 (평균 문자 신뢰도)
RETRY_AVG_CONF = 0.60

# 신뢰도 로그 파일
CONFIDENCE_LOG_FILE = "plate_history/confidence_log.json"

# 답지 파일 경로
ANSWER_FILE = "plate_history/answer.txt"

# 답지 매칭 완료 로그 파일 (중복 저장 방지용)
ANSWER_MATCHED_LOG_FILE = "plate_history/answer_matched_log.json"

# 신뢰도 비교 함수
def compare_character_confidence(new_chars, new_confs, existing_chars, existing_confs):
    """
    각 문자별 신뢰도를 비교하여 전체적인 우위를 판단
    반환: (새 결과가 우위인지, 승리한 문자 수, 총 문자 수)
    """
    if not new_chars or not existing_chars:
        return False, 0, 0
    
    # 문자 매칭 (위치 기반)
    min_len = min(len(new_chars), len(existing_chars))
    new_wins = 0
    total_compared = 0
    
    for i in range(min_len):
        if new_chars[i] == existing_chars[i]:
            # 같은 문자면 신뢰도 비교
            if new_confs[i] > existing_confs[i]:
                new_wins += 1
            total_compared += 1
        else:
            # 다른 문자면 신뢰도가 높은 쪽이 우위
            if new_confs[i] > existing_confs[i]:
                new_wins += 1
            total_compared += 1
    
    # 남은 문자들 처리
    if len(new_chars) > min_len:
        # 새 결과에 더 많은 문자가 있으면 추가 승점
        new_wins += len(new_chars) - min_len
        total_compared += len(new_chars) - min_len
    elif len(existing_chars) > min_len:
        # 기존 결과에 더 많은 문자가 있으면 기존 우위
        total_compared += len(existing_chars) - min_len
    
    # 60% 이상 승리하면 새 결과가 우위
    is_new_better = (new_wins / total_compared) >= 0.6 if total_compared > 0 else False
    
    return is_new_better, new_wins, total_compared

# 지역명 정규화 함수
def normalize_region_name(text):
    for i in range(len(text) - 1):
        if re.match(r'[가-힣]', text[i]) and re.match(r'[가-힣]', text[i+1]):
            region_pair = text[i] + text[i+1]
            for standard_region in region_names:
                if region_pair == standard_region or region_pair == standard_region[::-1]:
                    return standard_region
    return None

# 번호판 방향 감지 및 텍스트 재정렬 함수
def detect_plate_orientation_and_reorder(detected_chars, plate_crop):
    if not detected_chars or len(detected_chars) < 3:  # 최소 3자 이상으로 변경
        return []
    height, width = plate_crop.shape[:2]
    aspect_ratio = width / height
    
    # 3개 이상의 문자가 감지되면 지그재그 2줄 알고리즘 사용
    if len(detected_chars) >= 3:
        print(f"[방향감지] {len(detected_chars)}개 문자 감지 → 지그재그 2줄 알고리즘 적용")
        return sort_two_line_plate_zigzag(detected_chars, plate_crop)
    elif aspect_ratio > 1.8:
        print(f"[방향감지] 가로세로비 {aspect_ratio:.2f} → 기존 2줄 알고리즘 적용")
        return sort_two_line_plate_advanced(detected_chars, plate_crop)
    else:
        print(f"[방향감지] 1줄 번호판으로 처리")
        return sorted(detected_chars, key=lambda x: x[0])

# 2줄 번호판 정렬 함수 (지그재그 형태 및 상단 박스 크기 검증 포함)
def sort_two_line_plate_simple(detected_chars, plate_crop):
    if not detected_chars or len(detected_chars) < 4:
        return []
    
    height, width = plate_crop.shape[:2]
    y_coords = [char[1] for char in detected_chars]
    y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
    
    # y좌표 분산이 작으면 1줄로 간주
    if y_variance < 200:
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 2줄 번호판으로 간주하고 지그재그 형태 검증
    y_mean = np.mean(y_coords)
    top_line = [char for char in detected_chars if char[1] <= y_mean]
    bottom_line = [char for char in detected_chars if char[1] > y_mean]
    
    # 각 줄의 문자 수가 너무 적으면 1줄로 간주
    if len(top_line) < 2 or len(bottom_line) < 2:
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 상단 줄과 하단 줄의 평균 박스 크기 계산
    def calculate_line_box_sizes(line_chars):
        if not line_chars:
            return 0, 0
        widths = []
        heights = []
        for char in line_chars:
            # char[0]은 center_x, char[1]은 center_y
            # 실제 박스 크기는 추정 (문자 간격 기준)
            widths.append(1)  # 기본값
            heights.append(1)  # 기본값
        return np.mean(widths), np.mean(heights)
    
    top_width, top_height = calculate_line_box_sizes(top_line)
    bottom_width, bottom_height = calculate_line_box_sizes(bottom_line)
    
    # 상단 박스가 더 작아야 함 (y좌표 기준으로 판단)
    top_smaller = np.mean([char[1] for char in top_line]) < np.mean([char[1] for char in bottom_line])
    
    print(f"[2줄 검증] 상단: {len(top_line)}개, 하단: {len(bottom_line)}개")
    print(f"[2줄 검증] 상단 작음: {top_smaller}")
    
    # 상단이 작은 경우에만 2줄로 처리 (x좌표 겹침 허용)
    if top_smaller:
        # 각 줄을 좌에서 우로 정렬
        top_line.sort(key=lambda x: x[0])
        bottom_line.sort(key=lambda x: x[0])
        
        # 첫째줄(상단)부터 시작하여 지그재그 형태로 정렬
        result = []
        top_idx = 0
        bottom_idx = 0
        
        # 상단과 하단을 번갈아가며 배치 (지그재그)
        while top_idx < len(top_line) and bottom_idx < len(bottom_line):
            if top_idx < len(top_line):
                result.append(top_line[top_idx])
                top_idx += 1
            if bottom_idx < len(bottom_line):
                result.append(bottom_line[bottom_idx])
                bottom_idx += 1
        
        # 남은 문자들 추가
        result.extend(top_line[top_idx:])
        result.extend(bottom_line[bottom_idx:])
        
        print(f"[2줄 검증] 지그재그 정렬 완료: {len(result)}개 문자")
        return result
    else:
        # 2줄 조건을 만족하지 않으면 1줄로 처리
        if not top_smaller:
            print(f"[2줄 검증] 2줄 조건 불만족: 상단이 하단보다 큼 → 1줄로 처리")
        else:
            print(f"[2줄 검증] 2줄 조건 불만족: 기타 이유 → 1줄로 처리")
        return sorted(detected_chars, key=lambda x: x[0])

# 2줄 번호판 정렬 함수 (개선된 버전)
def sort_two_line_plate_advanced(detected_chars, plate_crop):
    if not detected_chars or len(detected_chars) < 4:
        return []
    
    height, width = plate_crop.shape[:2]
    
    # 문자들을 y좌표로 그룹화
    y_coords = [char[1] for char in detected_chars]
    y_mean = np.mean(y_coords)
    
    # 상단/하단 줄 분리
    top_line = [char for char in detected_chars if char[1] <= y_mean]
    bottom_line = [char for char in detected_chars if char[1] > y_mean]
    
    # 각 줄의 문자 수가 적절한지 확인
    if len(top_line) < 2 or len(bottom_line) < 2:
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 상단 박스가 더 작은지 확인 (y좌표 기준)
    top_avg_y = np.mean([char[1] for char in top_line])
    bottom_avg_y = np.mean([char[1] for char in bottom_line])
    top_smaller = top_avg_y < bottom_avg_y
    
    print(f"[2줄 검증] 상단: {len(top_line)}개 (y평균: {top_avg_y:.1f}), 하단: {len(bottom_line)}개 (y평균: {bottom_avg_y:.1f})")
    print(f"[2줄 검증] 상단 작음: {top_smaller}")
    
    # 상단이 작은 경우에만 2줄로 처리 (x좌표 겹침 허용)
    if top_smaller:
        # 각 줄을 좌에서 우로 정렬
        top_line.sort(key=lambda x: x[0])
        bottom_line.sort(key=lambda x: x[0])
        
        # 첫째줄(상단)부터 시작하여 지그재그 형태로 정렬
        result = []
        max_chars = max(len(top_line), len(bottom_line))
        
        for i in range(max_chars):
            if i < len(top_line):
                result.append(top_line[i])
            if i < len(bottom_line):
                result.append(bottom_line[i])
        
        print(f"[2줄 검증] 지그재그 정렬 완료: {len(result)}개 문자")
        return result
    else:
        # 2줄 조건을 만족하지 않으면 1줄로 처리
        if not top_smaller:
            print(f"[2줄 검증] 2줄 조건 불만족: 상단이 하단보다 큼 → 1줄로 처리")
        else:
            print(f"[2줄 검증] 2줄 조건 불만족: 기타 이유 → 1줄로 처리")
        return sorted(detected_chars, key=lambda x: x[0])

def sort_two_line_plate_zigzag(detected_chars, plate_crop):
    """
    3개 이상의 문자가 지그재그 형태로 감지될 때 상단부터 좌에서 우로 출력하는 2줄 알고리즘
    """
    if not detected_chars or len(detected_chars) < 3:
        return []
    
    height, width = plate_crop.shape[:2]
    
    # 문자들을 y좌표로 그룹화하여 상단/하단 줄 분리
    y_coords = [char[1] for char in detected_chars]
    y_mean = np.mean(y_coords)
    
    # 상단/하단 줄 분리
    top_line = [char for char in detected_chars if char[1] <= y_mean]
    bottom_line = [char for char in detected_chars if char[1] > y_mean]
    
    print(f"[지그재그 2줄] 전체: {len(detected_chars)}개, 상단: {len(top_line)}개, 하단: {len(bottom_line)}개")
    
    # 각 줄의 문자 수가 적절한지 확인 (최소 1개 이상)
    if len(top_line) < 1 or len(bottom_line) < 1:
        print(f"[지그재그 2줄] 줄별 문자 수 부족 → 1줄로 처리")
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 상단과 하단의 y좌표 차이 계산
    top_avg_y = np.mean([char[1] for char in top_line])
    bottom_avg_y = np.mean([char[1] for char in bottom_line])
    y_diff = abs(bottom_avg_y - top_avg_y)
    
    # y좌표 차이가 너무 작으면 1줄로 처리
    min_y_diff = height * 0.1  # 이미지 높이의 10% 이상 차이가 있어야 2줄로 인식
    if y_diff < min_y_diff:
        print(f"[지그재그 2줄] y좌표 차이 부족 ({y_diff:.1f} < {min_y_diff:.1f}) → 1줄로 처리")
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 각 줄을 좌에서 우로 정렬
    top_line.sort(key=lambda x: x[0])
    bottom_line.sort(key=lambda x: x[0])
    
    print(f"[지그재그 2줄] 상단 줄: {[f'({char[0]:.0f},{char[1]:.0f})' for char in top_line]}")
    print(f"[지그재그 2줄] 하단 줄: {[f'({char[0]:.0f},{char[1]:.0f})' for char in bottom_line]}")
    
    # 상단부터 시작하여 지그재그 형태로 정렬
    result = []
    max_chars = max(len(top_line), len(bottom_line))
    
    for i in range(max_chars):
        # 상단 줄의 문자 추가
        if i < len(top_line):
            result.append(top_line[i])
        # 하단 줄의 문자 추가
        if i < len(bottom_line):
            result.append(bottom_line[i])
    
    print(f"[지그재그 2줄] 지그재그 정렬 완료: {len(result)}개 문자")
    print(f"[지그재그 2줄] 최종 순서: {[f'({char[0]:.0f},{char[1]:.0f})' for char in result]}")
    
    return result

def sort_two_line_plate_improved(detected_chars, plate_crop):
    """
    개선된 2줄 번호판 정렬 알고리즘
    - 3개 이상 문자 감지 시 자동으로 2줄 모드 활성화
    - 상단부터 좌에서 우로 지그재그 출력
    - 더 정확한 줄 구분을 위한 클러스터링 적용
    """
    if not detected_chars or len(detected_chars) < 3:
        return []
    
    height, width = plate_crop.shape[:2]
    
    # 문자들의 y좌표를 기준으로 클러스터링하여 2줄 구분
    y_coords = np.array([char[1] for char in detected_chars])
    
    # K-means 클러스터링으로 2줄 구분 (더 정확한 분리)
    from sklearn.cluster import KMeans
    
    try:
        # y좌표를 2D 배열로 변환
        y_coords_2d = y_coords.reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(y_coords_2d)
        
        # 클러스터별로 문자 분리
        cluster_0 = [char for i, char in enumerate(detected_chars) if cluster_labels[i] == 0]
        cluster_1 = [char for i, char in enumerate(detected_chars) if cluster_labels[i] == 1]
        
        # y좌표 평균이 작은 것을 상단 줄로 설정
        if np.mean([char[1] for char in cluster_0]) < np.mean([char[1] for char in cluster_1]):
            top_line = cluster_0
            bottom_line = cluster_1
        else:
            top_line = cluster_1
            bottom_line = cluster_0
            
    except Exception as e:
        print(f"[개선된 2줄] K-means 클러스터링 실패: {e} → 기본 방법 사용")
        # 기본 방법으로 fallback
        y_mean = np.mean(y_coords)
        top_line = [char for char in detected_chars if char[1] <= y_mean]
        bottom_line = [char for char in detected_chars if char[1] > y_mean]
    
    print(f"[개선된 2줄] 전체: {len(detected_chars)}개, 상단: {len(top_line)}개, 하단: {len(bottom_line)}개")
    
    # 각 줄의 문자 수가 적절한지 확인
    if len(top_line) < 2 or len(bottom_line) < 2:
        print(f"[개선된 2줄] 줄별 문자 수 부족 → 1줄로 처리")
        return sorted(detected_chars, key=lambda x: x[0])
    
    # 각 줄을 좌에서 우로 정렬
    top_line.sort(key=lambda x: x[0])
    bottom_line.sort(key=lambda x: x[0])
    
    # 상단부터 시작하여 지그재그 형태로 정렬
    result = []
    max_chars = max(len(top_line), len(bottom_line))
    
    for i in range(max_chars):
        # 상단 줄의 문자 추가
        if i < len(top_line):
            result.append(top_line[i])
        # 하단 줄의 문자 추가
        if i < len(bottom_line):
            result.append(bottom_line[i])
    
    print(f"[개선된 2줄] 지그재그 정렬 완료: {len(result)}개 문자")
    return result

# 번호판 텍스트 정규화 함수
def normalize_plate_text(text, plate_crop):
    if not text or len(text) < 4:
        return ""
    
    original_text = text
    print(f"[정규화] 원본 텍스트: '{original_text}'")
    
    # 2줄 번호판에서 하이픈(-) 제거
    if '-' in text:
        text = text.replace('-', '')
        print(f"[정규화] 2줄 번호판 하이픈 제거: '{original_text}' → '{text}'")
    
    # 한글 개수 확인 및 "영" 글자 제거
    korean_chars = [char for char in text if re.match(r'[가-힣]', char)]
    if len(korean_chars) == 4 and '영' in text:
        text = text.replace('영', '')
        print(f"[정규화] 한글 4개 중 '영' 제거: '{original_text}' → '{text}'")
    
    normalized_region = normalize_region_name(text)
    if normalized_region:
        print(f"[정규화] 지역명 감지: '{normalized_region}'")
        for i in range(len(text) - 1):
            if re.match(r'[가-힣]', text[i]) and re.match(r'[가-힣]', text[i+1]):
                region_pair = text[i] + text[i+1]
                if region_pair == normalized_region or region_pair == normalized_region[::-1]:
                    if region_pair != normalized_region:
                        print(f"[정규화] 지역명 역순 수정: '{region_pair}' → '{normalized_region}'")
                    text = text[:i] + normalized_region + text[i+2:]
                    break
    else:
        print(f"[정규화] 지역명 미감지 - 원본 유지")
    
    print(f"[정규화] 최종 결과: '{original_text}' → '{text}'")
    return text

# 한글 텍스트 출력 함수
def put_text_with_pil(img, text, position, font_size):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_paths = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/batang.ttc",
        "C:/Windows/Fonts/gulim.ttc",
        None
    ]
    
    font = None
    for font_path in font_paths:
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    x, y = position
    draw.text((x, y), text, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(255, 255, 255))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 번호판 탐지 함수
def detect_plate(frame, plate_model, target_size=(320, 320)):
    frame = cv2.resize(frame, target_size)
    plate_results = plate_model(frame, conf=0.25, verbose=False, half=True)
    return plate_results, frame

# 2차 검증 함수 (같은 letter 모델 사용)
def verify_with_letter_model(plate_crop, letter_model, classes):
    plate_crop_2nd = cv2.convertScaleAbs(plate_crop, alpha=1.2, beta=30)
    # 스케일 업으로 글자 확대 효과
    h2, w2 = plate_crop_2nd.shape[:2]
    scale_factor = 1.3
    plate_crop_2nd = cv2.resize(plate_crop_2nd, (int(w2 * scale_factor), int(h2 * scale_factor)))
    letter_results = letter_model(plate_crop_2nd, conf=0.15, verbose=False, half=True)
    detected_chars = []
    confidence_log = []
    conf_values = []
    
    for box in letter_results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls < len(classes):
            char = classes[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detected_chars.append((center_x, center_y, char))
            confidence_log.append(f"{char}({conf:.2f})")
            conf_values.append(conf)
    
    reordered_chars = detect_plate_orientation_and_reorder(detected_chars, plate_crop)
    if not reordered_chars:
        return "", 0.0, [], []
    
    # 2줄 번호판 판정 및 처리 (2차 검수)
    height, width = plate_crop.shape[:2]
    aspect_ratio = width / height
    
    if aspect_ratio > 1.8 and len(reordered_chars) >= 8:  # 2줄 번호판 가능성
        # y좌표로 상단/하단 분리
        y_coords = [char[1] for char in reordered_chars]
        y_mean = np.mean(y_coords)
        top_line = [char for char in reordered_chars if char[1] <= y_mean]
        bottom_line = [char for char in reordered_chars if char[1] > y_mean]
        
        if len(top_line) >= 3 and len(bottom_line) >= 3:  # 2줄 번호판으로 확정
            # 각 줄을 x좌표로 정렬
            top_line.sort(key=lambda x: x[0])
            bottom_line.sort(key=lambda x: x[0])
            
            # 2줄 번호판 형식으로 텍스트 구성
            top_text = ''.join([char for _, _, char in top_line])
            bottom_text = ''.join([char for _, _, char in bottom_line])
            text = f"{top_text}-{bottom_text}"
            print(f"[2차 검수-2줄] 상단: '{top_text}', 하단: '{bottom_text}' → '{text}'")
        else:
            # 2줄 조건 불만족 시 1줄로 처리
            text = ''.join([char for _, _, char in reordered_chars]).replace(' ', '')
            print(f"[2차 검수-2줄] 조건 불만족 → 1줄로 처리: '{text}'")
    else:
        # 1줄 번호판으로 처리
        text = ''.join([char for _, _, char in reordered_chars]).replace(' ', '')
        print(f"[2차 검수-1줄] 처리: '{text}'")
    normalized_text = normalize_plate_text(text, plate_crop)
    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
    

    
    if confidence_log:
        print(f"[2차 검수] 신뢰도: {' '.join(confidence_log)} | 평균 {avg_conf:.2f}")
    
    return normalized_text, avg_conf, detected_chars, conf_values

# 번호판 내용 읽기 함수
def read_plate_content(plate_crop, letter_model, classes, txt_file):
    if plate_crop.shape[0] < 30 or plate_crop.shape[1] < 80:
        return "", 0.0, [], []
    
    plate_crop = cv2.convertScaleAbs(plate_crop, alpha=1.5, beta=50)
    letter_results = letter_model(plate_crop, conf=0.2, verbose=False, half=False)
    detected_chars = []
    confidence_log = []
    conf_values = []
    
    for box in letter_results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        if cls < len(classes):
            char = classes[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            detected_chars.append((center_x, center_y, char))
            confidence_log.append(f"{char}({conf:.2f})")
            conf_values.append(conf)
    
    reordered_chars = detect_plate_orientation_and_reorder(detected_chars, plate_crop)
    if not reordered_chars:
        return "", 0.0, [], []
    
    # 2줄 번호판 판정 및 처리
    height, width = plate_crop.shape[:2]
    aspect_ratio = width / height
    
    # detect_plate_orientation_and_reorder에서 이미 2줄 처리가 완료되었는지 확인
    # 지그재그 2줄 알고리즘이 적용된 경우, 결과를 그대로 사용
    if len(reordered_chars) >= 3:
        # y좌표로 상단/하단 분리하여 2줄 여부 재확인
        y_coords = [char[1] for char in reordered_chars]
        y_mean = np.mean(y_coords)
        top_line = [char for char in reordered_chars if char[1] <= y_mean]
        bottom_line = [char for char in reordered_chars if char[1] > y_mean]
        
        # 각 줄에 최소 1개 이상의 문자가 있고, y좌표 차이가 충분한 경우 2줄로 처리
        min_y_diff = height * 0.1  # 이미지 높이의 10% 이상 차이
        if (len(top_line) >= 1 and len(bottom_line) >= 1 and 
            abs(np.mean([char[1] for char in bottom_line]) - np.mean([char[1] for char in top_line])) >= min_y_diff):
            
            # 2줄 번호판 형식으로 텍스트 구성
            top_text = ''.join([char for _, _, char in top_line])
            bottom_text = ''.join([char for _, _, char in bottom_line])
            text = f"{top_text}-{bottom_text}"
            print(f"[2줄 번호판] 상단: '{top_text}', 하단: '{bottom_text}' → '{text}'")
        else:
            # 2줄 조건 불만족 시 1줄로 처리
            text = ''.join([char for _, _, char in reordered_chars]).replace(' ', '')
            print(f"[2줄 판정] 조건 불만족 → 1줄로 처리: '{text}'")
    else:
        # 1줄 번호판으로 처리
        text = ''.join([char for _, _, char in reordered_chars]).replace(' ', '')
        print(f"[1줄 번호판] 처리: '{text}'")
    normalized_text = normalize_plate_text(text, plate_crop)
    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
    
    if confidence_log:
        print(f"[1차 검수] 신뢰도: {' '.join(confidence_log)} | 평균 {avg_conf:.2f}")
    

    
    korean_plate_chars = get_korean_plate_chars(classes)
    if not is_valid_plate(normalized_text, korean_plate_chars):
        return "", 0.0, [], []
    
    # 평균 신뢰도가 낮으면 2차 재인식 시도
    if avg_conf < RETRY_AVG_CONF:
        print(f"[재시도] 평균 신뢰도 {avg_conf:.2f} < {RETRY_AVG_CONF:.2f} → 2차 재인식 수행")
        second_text, second_avg, _, _ = verify_with_letter_model(plate_crop, letter_model, classes)
        if second_text and is_valid_plate(second_text, korean_plate_chars):
            print(f"[재시도 결과] 2차 평균 {second_avg:.2f}")
            # 2차가 더 높거나 임계값을 넘으면 교체
            if second_avg >= avg_conf or second_avg >= RETRY_AVG_CONF:
                normalized_text = second_text
                avg_conf = second_avg
            else:
                # 신뢰도 낮으면 스킵
                print(f"[재시도 결과] 신뢰도 개선 없음 → 스킵")
                return "", 0.0, [], []
        else:
            print(f"[재시도 결과] 실패 또는 유효하지 않은 결과 → 스킵")
            return "", 0.0, [], []
    
    if normalized_text and os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            existing_plates = [line.strip() for line in f if line.strip()]
        
        print(f"[현재 감지] '{normalized_text}'")
        if existing_plates:
            print(f"[기존 번호판] {', '.join(existing_plates)}")
        else:
            print("[기존 번호판] 없음")
        
        current_last_4 = normalized_text[-4:] if len(normalized_text) >= 4 else ""
        print(f"[중복 검사] 현재 번호판 '{normalized_text}' 뒤 4자리: '{current_last_4}'")
        
        # 기존 번호판들의 뒤 4자리 확인
        matching_plates = []
        for plate in existing_plates:
            if len(plate) >= 4:
                plate_last_4 = plate[-4:]
                if current_last_4 == plate_last_4:
                    matching_plates.append(plate)
        
        if matching_plates:
            print(f"[중복 감지] 뒤 4자리 '{current_last_4}' 중복 발견: {', '.join(matching_plates)}")
            print(f"[중복 감지] 2차 검증 시작")
            second_text, second_avg, _, _ = verify_with_letter_model(plate_crop, letter_model, classes)
            if second_text and is_valid_plate(second_text, korean_plate_chars):
                second_last_4 = second_text[-4:] if len(second_text) >= 4 else ""
                print(f"[2차 검증] 결과: '{second_text}' (뒤 4자리: '{second_last_4}')")
                if second_last_4 == current_last_4:
                    print(f"[2차 검증] 동일한 결과 '{second_text}', 중복으로 제외")
                    return "", 0.0, [], []
                print(f"[2차 검증] 다른 결과 발견: '{second_text}' (원본: '{normalized_text}')")
                return second_text, 0.0, [], []
            print(f"[2차 검증] 실패 또는 유효하지 않은 결과")
            return "", 0.0, [], []
        else:
            print(f"[중복 검사] 중복 없음 - 저장 진행")
    elif normalized_text:
        print(f"[새 번호판] '{normalized_text}' (첫 번째 감지)")
    
    return normalized_text, avg_conf, reordered_chars, conf_values

# 한국 번호판 형식 검증 함수
def is_valid_plate(text, korean_plate_chars):
    if not text or len(text) < 7:
        return False
    

    
    # 2줄 번호판 형식 검증 (하이픈으로 구분)
    if '-' in text:
        parts = text.split('-')
        if len(parts) != 2:
            return False
        
        top_part, bottom_part = parts
        # 상단: 지역명(2자) + 숫자(2-3자) + 한글(1자)
        # 하단: 숫자(4자)
        pattern_top = rf'^[{korean_plate_chars}]{{2}}\d{{2,3}}[{korean_plate_chars}]$'
        pattern_bottom = r'^\d{4}$'
        
        is_valid = bool(re.match(pattern_top, top_part) and re.match(pattern_bottom, bottom_part))
        
        if is_valid and re.match(rf'^[{korean_plate_chars}]{{2}}', top_part):
            region = top_part[:2]
            if region not in region_names:
                return False
        return is_valid
    
    # 1줄 번호판 형식 검증
    pattern_region = rf'^[{korean_plate_chars}]{{2}}\d{{2,3}}[{korean_plate_chars}]\d{{4}}$'
    pattern_basic = rf'^\d{{2,3}}[{korean_plate_chars}]\d{{4}}$'
    is_valid = bool(re.match(pattern_region, text) or re.match(pattern_basic, text))
    
    if is_valid and re.match(rf'^[{korean_plate_chars}]{{2}}', text):
        region = text[:2]
        if region not in region_names:
            return False
    return is_valid

# 번호판 중복 체크 함수 (신뢰도 기반)
def is_duplicate_plate(new_text, existing_texts, korean_plate_chars, new_confidence=0.0):
    if not new_text or not is_valid_plate(new_text, korean_plate_chars):
        return False, None
    
    match = re.match(rf'^([{korean_plate_chars}]{{2}})?\d{{2,3}}([{korean_plate_chars}])(\d{{4}})$', new_text)
    if not match:
        return False, None
    
    _, korean_char, last_digits = match.groups()
    new_key = f"{korean_char}{last_digits}"
    
    for existing_text in existing_texts:
        match = re.match(rf'^([{korean_plate_chars}]{{2}})?\d{{1,3}}([{korean_plate_chars}])(\d{{4}})$', existing_text)
        if match:
            _, existing_korean_char, existing_last_digits = match.groups()
            existing_key = f"{existing_korean_char}{existing_last_digits}"
            if new_key == existing_key:
                return True, existing_text
    return False, None

# 히스토리 이미지 렌더링 함수
def render_history(history, frame_height, history_width=200, max_history=3):
    history_img = np.zeros((frame_height, history_width, 3), dtype=np.uint8)
    history_img.fill(240)
    
    if not history:
        history_img = cv2.putText(history_img, "No plates", 
                                 (10, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return history_img
    
    recent_history = history[-max_history:]
    item_height = frame_height // max_history
    
    for i, (filename, text) in enumerate(reversed(recent_history)):
        try:
            img = cv2.imread(filename)
            if img is not None:
                h, w = img.shape[:2]
                aspect_ratio = w / h
                
                target_height = item_height - 10
                target_width = int(target_height * aspect_ratio)
                
                if target_width > history_width - 10:
                    target_width = history_width - 10
                    target_height = int(target_width / aspect_ratio)
                
                img_resized = cv2.resize(img, (target_width, target_height))
                
                y_start = i * item_height + 5
                x_start = (history_width - target_width) // 2
                
                history_img[y_start:y_start + target_height, 
                           x_start:x_start + target_width] = img_resized
                
                text_y = y_start + target_height + 10
                if text_y < frame_height - 10:
                    history_img = put_text_with_pil(history_img, text, 
                                                  ((history_width - target_width) // 2, text_y), 12)
        except:
            y_start = i * item_height + 5
            cv2.putText(history_img, f"Error: {text}", (5, y_start + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    return history_img

# 답지 로드 함수
def load_answer_list():
    """답지 파일에서 정답 번호판 목록을 로드하는 함수"""
    answer_list = []
    if os.path.exists(ANSWER_FILE):
        try:
            with open(ANSWER_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 뒤 4자리 숫자 추출
                        if len(line) >= 4:
                            last_4 = line[-4:]
                            if last_4.isdigit():
                                answer_list.append({
                                    'full_text': line,
                                    'last_4': last_4
                                })
            print(f"[답지] {len(answer_list)}개의 정답 번호판 로드됨")
        except Exception as e:
            print(f"[답지] 답지 로드 실패: {e}")
    else:
        print(f"[답지] 답지 파일이 없습니다: {ANSWER_FILE}")
    return answer_list

# 신뢰도 로그 로드 함수
def load_confidence_log():
    if os.path.exists(CONFIDENCE_LOG_FILE):
        try:
            with open(CONFIDENCE_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

# 신뢰도 로그 저장 함수
def save_confidence_log(confidence_data):
    try:
        with open(CONFIDENCE_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(confidence_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Main] 신뢰도 로그 저장 실패: {e}")

# 답지 매칭 완료 로그 로드 함수
def load_answer_matched_log():
    if os.path.exists(ANSWER_MATCHED_LOG_FILE):
        try:
            with open(ANSWER_MATCHED_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

# 답지 매칭 완료 로그 저장 함수
def save_answer_matched_log(matched_data):
    try:
        with open(ANSWER_MATCHED_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Main] 답지 매칭 로그 저장 실패: {e}")

# TXT 파일에 번호판 번호 저장 (문자별 신뢰도 기반)
def save_to_txt(text, txt_file, korean_plate_chars, new_chars=None, new_confs=None):
    existing_texts = []
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            existing_texts = [line.strip() for line in f if line.strip()]
    
    # 답지 로드
    answer_list = load_answer_list()
    
    # 신뢰도 로그 로드
    confidence_log = load_confidence_log()
    
    # 현재 번호판의 키 생성 (뒤 4자리 기준)
    current_last_4 = text[-4:] if len(text) >= 4 else ""
    current_key = current_last_4
    
    # 답지와 비교하여 정답 여부 확인
    is_correct_answer = False
    matched_answer = None
    if current_last_4 and current_last_4.isdigit():
        for answer in answer_list:
            if answer['last_4'] == current_last_4:
                is_correct_answer = True
                matched_answer = answer['full_text']
                break
    
    if is_correct_answer:
        # 답지 매칭 완료 로그 확인
        answer_matched_log = load_answer_matched_log()
        
        # 이미 매칭 완료된 답지인지 확인
        if current_last_4 in answer_matched_log:
            print(f"[답지] 이미 매칭 완료된 답지: '{text}' (뒤 4자리: {current_last_4}) - 저장 건너뜀")
            return  # 중복 저장 방지
        
        # 정답인 경우 히스토리에 업로드하고 폴더에 저장
        print(f"[답지] 정답 발견! '{text}' (뒤 4자리: {current_last_4}) → '{matched_answer}'")
        
        # 정답 번호판을 별도 파일에 저장
        correct_answer_file = os.path.join(os.path.dirname(txt_file), "correct_answers.txt")
        with open(correct_answer_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {text} | 정답: {matched_answer}\n")
        
        # 정답 히스토리 파일에도 저장
        history_file = os.path.join(os.path.dirname(txt_file), "answer_history.txt")
        with open(history_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {text} | 정답: {matched_answer} | 신뢰도: {np.mean(new_confs) if new_confs else 0.0:.2f}\n")
        
        # 답지 매칭 완료 로그에 추가
        answer_matched_log[current_last_4] = {
            "detected_text": text,
            "answer_text": matched_answer,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": float(np.mean(new_confs)) if new_confs else 0.0
        }
        save_answer_matched_log(answer_matched_log)
        
        print(f"[답지] 정답 번호판 '{text}' 저장 완료 (정답: {matched_answer}) - 중복 저장 방지 설정됨")
        
        # 답지에 매칭되는 번호판만 plate_numbers.txt에 저장
        # 기존 신뢰도 정보 확인
        existing_data = confidence_log.get(current_key, {"text": "", "chars": [], "confs": [], "avg_conf": 0.0})
        existing_text = existing_data.get("text", "")
        existing_chars = existing_data.get("chars", [])
        existing_confs = existing_data.get("confs", [])
        
        # 기존 번호판에서 같은 키를 가진 것 찾기
        if not existing_text:
            for plate in existing_texts:
                if len(plate) >= 4 and plate[-4:] == current_key:
                    existing_text = plate
                    break
        
        if not existing_text:
            # 새로운 번호판 - 저장 (답지에 매칭되는 경우 답지 이름 사용)
            save_text = matched_answer if matched_answer else text
            with open(txt_file, 'a', encoding='utf-8') as f:
                f.write(f"{save_text}\n")
            
            # 신뢰도 정보 저장 (답지에 매칭되는 경우 답지 이름 사용)
            confidence_log[current_key] = {
                "text": save_text,
                "chars": new_chars if new_chars else [],
                "confs": new_confs if new_confs else [],
                "avg_conf": float(np.mean(new_confs)) if new_confs else 0.0
            }
            save_confidence_log(confidence_log)
            print(f"[Main] 새 번호판 저장: {save_text}")
        else:
            # 중복 발견 - 문자별 신뢰도 비교
            if new_chars and new_confs and existing_chars and existing_confs:
                is_new_better, wins, total = compare_character_confidence(new_chars, new_confs, existing_chars, existing_confs)
                print(f"[Main] 중복 발견: 기존 '{existing_text}' vs 새 '{text}'")
                print(f"[Main] 문자별 비교: {wins}/{total} 승리 ({(wins/total*100):.1f}%)")
                
                if is_new_better:
                    # 새 결과가 우위 - 교체 (답지에 매칭되는 경우 답지 이름 사용)
                    save_text = matched_answer if matched_answer else text
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        for line in lines:
                            if line.strip() == existing_text:
                                f.write(f"{save_text}\n")
                                print(f"[Main] 문자별 신뢰도 우위 - 기존 번호판 교체: '{existing_text}' → '{save_text}'")
                            else:
                                f.write(line)
                    
                    # 신뢰도 정보 업데이트 (답지에 매칭되는 경우 답지 이름 사용)
                    confidence_log[current_key] = {
                        "text": save_text,
                        "chars": new_chars,
                        "confs": new_confs,
                        "avg_conf": float(np.mean(new_confs))
                    }
                    save_confidence_log(confidence_log)
                else:
                    print(f"[Main] 문자별 신뢰도 낮음 - 기존 번호판 유지")
            else:
                # 신뢰도 정보가 없으면 기존 로직 사용
                existing_avg = existing_data.get("avg_conf", 0.0)
                new_avg = float(np.mean(new_confs)) if new_confs else 0.0
                print(f"[Main] 중복 발견: 기존 '{existing_text}' (평균: {existing_avg:.2f}) vs 새 '{text}' (평균: {new_avg:.2f})")
                
                if new_avg > existing_avg:
                    # 평균 신뢰도가 높으면 교체 (답지에 매칭되는 경우 답지 이름 사용)
                    save_text = matched_answer if matched_answer else text
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        for line in lines:
                            if line.strip() == existing_text:
                                f.write(f"{save_text}\n")
                                print(f"[Main] 평균 신뢰도 높음 - 기존 번호판 교체: '{existing_text}' → '{save_text}'")
                            else:
                                f.write(line)
                    
                    confidence_log[current_key] = {
                        "text": save_text,
                        "chars": new_chars if new_chars else [],
                        "confs": new_confs if new_confs else [],
                        "avg_conf": new_avg
                    }
                    save_confidence_log(confidence_log)
                else:
                    print(f"[Main] 평균 신뢰도 낮음 - 기존 번호판 유지")
    else:
        # 답지에 매칭되지 않는 번호판은 저장하지 않음
        print(f"[답지] 답지에 없는 번호판: '{text}' (뒤 4자리: {current_last_4}) - plate_numbers.txt에 저장하지 않음")

# 카메라 연결 상태 확인 함수
def check_camera_connection(cap):
    if not cap.isOpened():
        print("❌ 카메라 연결이 끊어졌습니다.")
        return False
    return True

# 번호판 읽기 프로세스 함수
def read_process(plate_queue, history_queue, history_dir, letter_model_path, classes, txt_file):
    print("[Read Process] Starting YOLO letter detection...")
    try:
        letter_model = YOLO(letter_model_path)
        if torch.cuda.is_available():
            letter_model.to('cuda')
        print("[Read Process] YOLO letter model 초기화 완료")
        print("[Read Process] 2차 검수도 같은 모델로 진행합니다")
    except Exception as e:
        print(f"[Read Process] 모델 로딩 실패: {e}")
        return
    
    korean_plate_chars = get_korean_plate_chars(classes)
    processed_count = 0
    
    while True:
        try:
            data = plate_queue.get(timeout=1.0)
            if data is None:
                break
            plate_crop, px1, py1, px2, py2, frame = data
            if plate_crop is None or plate_crop.size == 0:
                continue
            
            read_start = time.time()
            text, confidence, chars, confs = read_plate_content(plate_crop, letter_model, classes, txt_file)
            read_time = int((time.time() - read_start) * 1000)
            
            if not text:
                continue
            
            print(f"[Read Process] 인식된 번호판: {text} (신뢰도: {confidence:.2f}, {read_time}ms)")
            history_queue.put((plate_crop, text, confidence, chars, confs, px1, py1, px2, py2, frame))
            
            processed_count += 1
            if processed_count % 30 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Read Process] 오류 발생: {e}")
            continue

# 메인 함수
def main():
    # CUDA 최적화 설정
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("[Main] CUDA 최적화 설정 완료")
    
    history_dir = "plate_history"
    txt_file = os.path.join(history_dir, "plate_numbers.txt")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    classes_file = "letterDetect_new/classes.txt"
    try:
        classes = load_classes(classes_file)
        korean_plate_chars = get_korean_plate_chars(classes)
        print(f"[Main] 클래스 로드 완료: {len(classes)}개 클래스")
    except Exception as e:
        print(f"[Main] classes.txt 로드 실패: {e}")
        return

    # 답지 로드
    answer_list = load_answer_list()

    letter_model_path = "letterDetect_new/train/exp/weights/best.pt"
    plate_model_path = "plateFolder/train/exp/weights/best.pt"

    if not os.path.exists(letter_model_path):
        print(f"❌ 글자 인식 모델 파일이 없습니다: {letter_model_path}")
        return
    if not os.path.exists(plate_model_path):
        print(f"❌ 번호판 검출 모델 파일이 없습니다: {plate_model_path}")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ USB 카메라에 연결할 수 없습니다.")
        return
    print(f"[Main] 카메라 설정: 해상도 640x480, 오토포커스 {'On' if cap.get(cv2.CAP_PROP_AUTOFOCUS) == 1 else 'Off'}")
    print(f"[Main] CUDA 사용 가능: {torch.cuda.is_available()}")

    t1 = time.time()
    plate_model = YOLO(plate_model_path)
    if torch.cuda.is_available():
        plate_model.to('cuda')
    t2 = time.time()
    print(f"[Main] 번호판 검출 모델 로딩: {int((t2 - t1) * 1000)} ms")

    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        # 컬러 필터 적용 (흑백 필터 제거)
        frame = cv2.resize(frame, (640, 480))
        _ = plate_model(frame, conf=0.3, max_det=3, verbose=False, half=False)
    else:
        print("❌ 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    t2 = time.time()
    print(f"[Main] CUDA warm-up: {int((t2 - t1) * 1000)} ms")

    TARGET_FPS = 20
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    LOG_INTERVAL = 2.0
    MEMORY_CLEANUP_INTERVAL = 4.0
    last_process_time = 0
    last_cleanup_time = 0
    last_history_update_time = 0
    frame_times = []
    frame_skip_counter = 0

    # 비디오 저장 초기화 (writer는 첫 프레임 크기에서 생성)
    video_path = os.path.join(history_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    video_writer = None

    plate_queue = Queue(maxsize=5)
    history_queue = Queue()
    read_proc = Process(target=read_process, args=(plate_queue, history_queue, history_dir, letter_model_path, classes, txt_file))
    read_proc.start()

    history = []
    consecutive_failures = 0

    while cap.isOpened():
        current_time = time.time()
        
        if not check_camera_connection(cap):
            consecutive_failures += 1
            if consecutive_failures > 10:
                print("❌ 카메라 연결 실패가 지속됩니다. 프로그램을 종료합니다.")
                break
            time.sleep(0.1)
            continue
        
        consecutive_failures = 0
        
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("프레임 읽기 실패. 종료합니다.")
            break

        # 컬러 필터 적용 (흑백 필터 제거)

        if current_time - last_process_time < FRAME_INTERVAL:
            frame_skip_counter += 1
            if frame_skip_counter % 3 != 0:
                continue
            else:
                frame_skip_counter = 0

        frame = cv2.resize(frame, (640, 480))
        output_img = frame.copy()
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        plate_results, resized_frame = detect_plate(frame, plate_model, target_size=(320, 320))

        frame_times.append(current_time)
        if len(frame_times) > 5:
            frame_times.pop(0)
        fps = len(frame_times) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0

        if current_time - last_process_time >= LOG_INTERVAL:
            print(f"[Main] FPS: {int(fps)}")
            last_process_time = current_time

        if current_time - last_cleanup_time >= MEMORY_CLEANUP_INTERVAL:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            last_cleanup_time = current_time

        read_status = "Off"
        if plate_results is not None:
            scale_x, scale_y = 640 / 320, 480 / 320
            for plate_result in plate_results:
                for plate_box in plate_result.boxes:
                    px1, py1, px2, py2 = map(int, [p * s for p, s in zip(plate_box.xyxy[0], [scale_x, scale_y, scale_x, scale_y])])
                    
                    plate_crop = frame[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue
                    
                    if plate_queue.full():
                        continue
                        
                    plate_queue.put((plate_crop, px1, py1, px2, py2, frame))
                    read_status = "On"

                    cv2.rectangle(output_img, (px1, py1), (px2, py2), (0, 255, 255), 2)

        if current_time - last_history_update_time >= 0.5:
            while not history_queue.empty():
                plate_crop, text, confidence, chars, confs, px1, py1, px2, py2, frame = history_queue.get()
                
                # 문자별 신뢰도 정보 추출
                char_list = [char for _, _, char in chars] if chars else []
                conf_list = confs if confs else []
                
                # 답지와 매칭되는지 먼저 확인
                answer_list = load_answer_list()
                current_last_4 = text[-4:] if len(text) >= 4 else ""
                matched_answer = None
                
                print(f"[Main] 답지 매칭 시도: '{text}' (뒤 4자리: '{current_last_4}')")
                print(f"[Main] 답지 목록: {[answer['last_4'] for answer in answer_list]}")
                
                if current_last_4 and current_last_4.isdigit():
                    for answer in answer_list:
                        if answer['last_4'] == current_last_4:
                            matched_answer = answer['full_text']
                            print(f"[Main] 답지 매칭 성공: '{current_last_4}' → '{matched_answer}'")
                            break
                
                # 답지에 매칭되는 번호판이 아니면 저장하지 않고 다음 항목으로 넘어감
                if not matched_answer:
                    print(f"[Main] 답지에 없는 번호판: '{text}' (뒤 4자리: {current_last_4}) - 저장 건너뜀")
                    continue
                
                # 답지와 매칭되는 경우 중복 저장 방지 확인
                if matched_answer:
                    answer_matched_log = load_answer_matched_log()
                    if current_last_4 in answer_matched_log:
                        print(f"[Main] 이미 매칭 완료된 답지: '{text}' (뒤 4자리: {current_last_4}) - 이미지 저장 건너뜀")
                        continue  # 이미 처리된 답지인 경우 건너뛰기
                
                # 중복 검사 (기존 방식 유지)
                is_duplicate, existing_text = is_duplicate_plate(text, [t for _, t in history], korean_plate_chars, confidence)
                
                if not is_duplicate:
                    # 답지와 매칭되는 경우 답지 이름으로 저장, 아닌 경우 감지된 텍스트로 저장
                    save_filename = matched_answer if matched_answer else text
                    filename = os.path.join(history_dir, f"{save_filename}.jpg")
                    cv2.imwrite(filename, plate_crop)
                    # 히스토리에 저장할 때도 답지에 매칭되는 경우 답지 이름 사용
                    save_text = matched_answer if matched_answer else text
                    history.append((filename, save_text))
                    if len(history) > 10:
                        old_filename = history[0][0]
                        try:
                            if os.path.exists(old_filename):
                                os.remove(old_filename)
                        except:
                            pass
                        history = history[1:]
                    
                    save_to_txt(text, txt_file, korean_plate_chars, char_list, conf_list)
                    output_img = put_text_with_pil(output_img, save_text, (px1, py1 - 25), 20)
                elif existing_text:
                    # 신뢰도 로그에서 기존 신뢰도 정보 확인
                    confidence_log = load_confidence_log()
                    current_last_4 = text[-4:] if len(text) >= 4 else ""
                    existing_data = confidence_log.get(current_last_4, {"chars": [], "confs": [], "avg_conf": 0.0})
                    existing_chars = existing_data.get("chars", [])
                    existing_confs = existing_data.get("confs", [])
                    existing_avg = existing_data.get("avg_conf", 0.0)
                    
                    # 문자별 신뢰도 비교
                    if char_list and conf_list and existing_chars and existing_confs:
                        is_new_better, wins, total = compare_character_confidence(char_list, conf_list, existing_chars, existing_confs)
                        print(f"[Main] 중복 발견: 기존 '{existing_text}' vs 새 '{text}'")
                        print(f"[Main] 문자별 비교: {wins}/{total} 승리 ({(wins/total*100):.1f}%)")
                        
                        if is_new_better:
                            # 문자별 신뢰도 우위 - 교체
                            print(f"[Main] 문자별 신뢰도 우위 - 기존 번호판 교체: '{existing_text}' → '{text}'")
                            # 기존 이미지 파일 삭제
                            old_filename = os.path.join(history_dir, f"{existing_text}.jpg")
                            try:
                                if os.path.exists(old_filename):
                                    os.remove(old_filename)
                            except:
                                pass
                            
                            # 새 이미지 저장 (답지와 매칭되는 경우 답지 이름으로 저장)
                            save_filename = matched_answer if matched_answer else text
                            filename = os.path.join(history_dir, f"{save_filename}.jpg")
                            cv2.imwrite(filename, plate_crop)
                            
                            # 히스토리에서 기존 항목 교체 (답지에 매칭되는 경우 답지 이름 사용)
                            save_text = matched_answer if matched_answer else text
                            for i, (hist_filename, hist_text) in enumerate(history):
                                if hist_text == existing_text:
                                    history[i] = (filename, save_text)
                                    break
                            
                            save_to_txt(text, txt_file, korean_plate_chars, char_list, conf_list)
                            output_img = put_text_with_pil(output_img, save_text, (px1, py1 - 25), 20)
                        else:
                            print(f"[Main] 문자별 신뢰도 낮음 - 기존 번호판 유지")
                    else:
                        # 평균 신뢰도 비교 (폴백)
                        if confidence > existing_avg:
                            print(f"[Main] 평균 신뢰도 높음 - 기존 번호판 교체: '{existing_text}' (평균: {existing_avg:.2f}) → '{text}' (평균: {confidence:.2f})")
                            # 기존 이미지 파일 삭제
                            old_filename = os.path.join(history_dir, f"{existing_text}.jpg")
                            try:
                                if os.path.exists(old_filename):
                                    os.remove(old_filename)
                            except:
                                pass
                            
                            # 새 이미지 저장 (답지와 매칭되는 경우 답지 이름으로 저장)
                            save_filename = matched_answer if matched_answer else text
                            filename = os.path.join(history_dir, f"{save_filename}.jpg")
                            cv2.imwrite(filename, plate_crop)
                            
                            # 히스토리에서 기존 항목 교체 (답지에 매칭되는 경우 답지 이름 사용)
                            save_text = matched_answer if matched_answer else text
                            for i, (hist_filename, hist_text) in enumerate(history):
                                if hist_text == existing_text:
                                    history[i] = (filename, save_text)
                                    break
                            
                            save_to_txt(text, txt_file, korean_plate_chars, char_list, conf_list)
                            output_img = put_text_with_pil(output_img, save_text, (px1, py1 - 25), 20)
                        else:
                            print(f"[Main] 평균 신뢰도 낮음 - 기존 번호판 유지 (기존: {existing_avg:.2f}, 새: {confidence:.2f})")
            last_history_update_time = current_time

        history_img = render_history(history, frame_height)
        output_img = np.hstack((output_img, history_img))

        output_img = put_text_with_pil(output_img, f"FPS: {int(fps)}", (10, 30), 28)
        output_img = put_text_with_pil(output_img, f"YOLO: On", (10, 60), 28)
        output_img = put_text_with_pil(output_img, f"Read: {read_status}", (10, 90), 28)

        cv2.imshow("번호판 검지 결과", output_img)

        # 비디오 라이터 생성 (최종 합성 프레임 기준, 폴백 포함)
        if video_writer is None:
            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, TARGET_FPS, (output_img.shape[1], output_img.shape[0]))
                if not video_writer.isOpened():
                    raise RuntimeError('mp4v open failed')
            except Exception:
                avi_path = os.path.splitext(video_path)[0] + '.avi'
                fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(avi_path, fourcc2, TARGET_FPS, (output_img.shape[1], output_img.shape[0]))
                video_path = avi_path

        # 비디오 프레임 저장
        if video_writer is not None:
            video_writer.write(output_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.time() - current_time
        if elapsed < FRAME_INTERVAL:
            time.sleep(FRAME_INTERVAL - elapsed)
        elif elapsed > FRAME_INTERVAL * 2:
            print(f"[Main] 프레임 처리 지연: {int(elapsed * 1000)}ms")

    try:
        plate_queue.put(None)
        read_proc.join(timeout=5.0)
        if read_proc.is_alive():
            read_proc.terminate()
            read_proc.join()
    except Exception as e:
        print(f"프로세스 종료 중 오류: {e}")
    finally:
        if 'video_writer' in locals() and video_writer is not None:
            try:
                video_writer.release()
                print(f"[Main] 비디오 저장 완료: {video_path}")
            except:
                pass
        cap.release()
        cv2.destroyAllWindows()

    print("\n=== 검지 히스토리 ===")
    for filename, text in history:
        print(f"파일: {filename}, 번호판: {text}")

if __name__ == "__main__":
    main()