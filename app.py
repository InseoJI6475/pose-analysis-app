# app.py (Cloudinary + Whitenoise + 환경 변수 최종 완성본)

from flask import Flask, render_template, request, jsonify
import os # <-- 환경 변수 읽기를 위해 필요
from werkzeug.utils import secure_filename
import analyzer_module
from whitenoise import WhiteNoise
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests # 동영상 다운로드용

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root="static/", prefix="static/")

# --- Cloudinary 설정 (★★★ Render 환경 변수에서 읽어오도록 수정 ★★★) ---
cloudinary.config(
  cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), # 환경 변수 CLOUDINARY_CLOUD_NAME 값 읽기
  api_key = os.environ.get("CLOUDINARY_API_KEY"),       # 환경 변수 CLOUDINARY_API_KEY 값 읽기
  api_secret = os.environ.get("CLOUDINARY_API_SECRET"), # 환경 변수 CLOUDINARY_API_SECRET 값 읽기
  secure = True
)

# 임시 파일 저장을 위한 폴더
TEMP_FOLDER = '/tmp'
if not os.path.exists(TEMP_FOLDER):
    try:
        os.makedirs(TEMP_FOLDER)
    except OSError as e:
        print(f"Could not create temp folder {TEMP_FOLDER}: {e}")
        TEMP_FOLDER = 'static/uploads'
        if not os.path.exists(TEMP_FOLDER):
             os.makedirs(TEMP_FOLDER)
app.config['UPLOAD_FOLDER'] = TEMP_FOLDER


# --- 전문가 지식 데이터베이스 (전체 - 이전과 동일) ---
EXPERT_KNOWLEDGE_BASE = {
    "back": {
        "흉·요추 후만 변형": "허리를 구부정하게 마는 자세는 **추간판(디스크) 후방으로 압력을 집중**시켜, 급성/만성 **추간판 탈출증(HIVD)** 및 **요추 염좌**의 발생 위험을 크게 높이는 가장 직접적인 원인입니다.",
        "골반 측방 붕괴": "한쪽으로 골반이 툭 빠지는 '트렌델렌버그 보행' 패턴은 고관절 안정성의 핵심인 **중둔근의 기능부전**을 의미합니다. 이는 연쇄적으로 **장경인대 증후군(IT Band Syndrome)**, **슬개대퇴통증 증후군(PFPS)**, 심지어 **족저근막염**까지 유발하는 '만악의 근원'이 될 수 있습니다.",
        "상체 과다 숙임": "군장 무게를 상쇄하기 위해 허리를 과도하게 숙이는 자세는, 허리뿐만 아니라 머리 무게를 지탱하기 위한 목과 등 상부 근육(상부 승모근, 견갑거근)의 과부하를 유발하여 **복합적인 만성 근막통증 증후군**과 **긴장성 두통**의 원인이 됩니다.",
        "요추 과전만 (Sway-back)": "배를 앞으로 내밀고 상체를 뒤로 젖힌 '스웨이백' 자세는 척추뼈 뒤쪽의 **척추후관절(Facet Joint)에 지속적인 압박과 마찰**을 가합니다. 이는 **척추후관절 증후군**, **척추관 협착증**, **척추분리증**의 위험을 증가시키는 매우 위험한 보상 패턴입니다.",
        "골반 후방 경사 (Flat Back)": "정상적인 허리 커브(전만)가 소실되어 허리가 편평해진 상태입니다. 이는 보행 시 지면의 충격을 척추가 제대로 흡수하지 못하게 만들어, **디스크의 퇴행성 변화를 가속화**하고 경미한 충격에도 허리 부상을 당하기 쉽게 만듭니다."
    },
    "neck": {
        "거북목 증후군": "**경추의 정상적인 C자 커브가 소실**되어 일자목 또는 역 C자 커브로 변형된 상태로, 머리가 1인치 앞으로 나갈 때마다 목뼈가 받는 하중은 약 4.5kg씩 기하급수적으로 증가합니다. **경추성 두통, 목 디스크, 흉곽출구 증후군**의 직접적인 원인이 되는 가장 흔하고 위험한 패턴입니다.",
        "굽은 등 보상 패턴": "목 자체의 문제라기보다, 굽은 등(흉추 후만)으로 인해 좁아진 시야를 확보하기 위한 2차적인 보상작용입니다. 근본 원인인 **등 상부의 가동성 저하와 날개뼈(견갑골)의 불안정성**을 해결하지 않으면 절대 완치되지 않습니다.",
        "전방 머리 자세": "거북목의 초기 단계로, 목과 어깨 주변 근육이 항상 불필요하게 긴장된 상태에 놓이게 됩니다. 이는 혈액순환을 방해하고 신경을 압박하여 **만성적인 어깨 결림, 팔 저림, 집중력 저하** 등을 유발할 수 있습니다.",
        "과도한 목 젖힘 (Military Neck)": "행군 후반부 피로가 쌓였을 때, 턱을 들고 목을 뒤로 젖히는 자세는 **경추 후관절에 심한 스트레스**를 주며, 목 뒤쪽 신경이 눌려 **급성 신경근병증(방사통)**을 유발할 수 있습니다."
    },
    "knee": {
        "대퇴사두근 우세 패턴": "계단을 내려오거나 착지할 때 무릎이 아프다면 이 패턴일 확률이 높습니다. 엉덩이 근육 대신 허벅지 앞쪽 근육을 과사용하여 **슬개골(무릎뼈)을 비정상적으로 압박**하고, 이는 **슬개대퇴통증 증후군(PFPS)** 및 **연골연화증**의 가장 흔한 원인이 됩니다.",
        "충격 흡수 부전": "무릎을 뻣뻣하게 편 상태로 걷는 '경직 보행'은 지면의 충격이 여과 없이 발목, 무릎, 고관절, 척추에 그대로 전달되게 만듭니다. 이는 단기적으로는 **정강이 통증(Shin Splint)**, 장기적으로는 **피로골절 및 퇴행성 관절염 악화**의 위험을 크게 증가시킵니다.",
        "과도한 무릎 굽힘": "보행 시 무릎을 과도하게 굽히는 것은 **햄스트링의 만성적인 단축이나 약화**를 의미할 수 있습니다. 이는 보행 효율을 급격히 떨어뜨려 쉽게 지치게 만들고, 무릎 관절에 불필요한 전단력(Shearing force)을 가합니다.",
        "무릎 과신전 (Genu Recurvatum)": "무릎이 '활처럼' 뒤로 꺾이는 자세는 무릎의 안정성을 담당하는 **후방십자인대(PCL)의 만성적인 이완**을 유발합니다. 이는 관절의 불안정성을 야기하여, 갑작스러운 방향 전환 시 심각한 무릎 부상으로 이어질 수 있습니다.",
        "무릎 내전 (Knee Valgus)": "보행 시 무릎이 안쪽으로 무너지는 패턴으로, **내측측부인대(MCL)의 스트레스**를 증가시키고 **전방십자인대(ACL) 파열**의 위험성을 높이는 매우 위험한 자세입니다. (주로 정면에서 관찰)"
    }
}
EXERCISE_PRESCRIPTION_DB = {
    "back": {
        "흉·요추 후만 변형": [
            {"type": "이완", "name": "흉추 신전 운동 (Cat-Cow Stretch)", "desc": "네발기기 자세에서 등과 허리를 위아래로 움직이며 뻣뻣한 등 상부의 가동성을 회복합니다.", "img": "static/exercises/cat_cow.gif"},
            {"type": "강화", "name": "코어 안정화 운동 (Bird-dog & Dead-bug)", "desc": "팔다리를 움직이는 동안에도 척추의 중립을 유지하는 훈련을 통해 심부 안정성을 강화합니다.", "img": "static/exercises/bird_dog.gif"}
        ],
        "골반 측방 붕괴": [
            {"type": "강화", "name": "중둔근 강화 운동 (Clamshell & Side Leg Raise)", "desc": "옆으로 누워 다리를 벌리거나 들어 올리며, 골반의 좌우 안정성을 담당하는 핵심 근육을 집중적으로 강화합니다.", "img": "static/exercises/clamshell.gif"},
            {"type": "강화", "name": "기능적 강화 운동 (Monster Walk & Single Leg Balance)", "desc": "밴드를 이용해 옆으로 걷거나 한 발로 서서, 실제 보행과 유사한 환경에서 고관절 안정성을 훈련합니다.", "img": "static/exercises/monster_walk.gif"}
        ],
        "요추 과전만 (Sway-back)": [
            {"type": "이완", "name": "고관절 굴곡근 스트레칭 (Hip Flexor Stretch)", "desc": "런지 자세에서 골반을 앞으로 밀어, 과도하게 긴장된 허리 앞쪽 근육과 허벅지 앞쪽을 이완시킵니다.", "img": "static/exercises/hip_flexor_stretch.jpg"},
            {"type": "강화", "name": "둔근 및 복근 강화 (Glute Bridge & Posterior Pelvic Tilt)", "desc": "누워서 엉덩이를 들거나 골반을 말아 올리며, 약화된 엉덩이/복근을 강화하여 허리의 과도한 커브를 줄여줍니다.", "img": "static/exercises/glute_bridge.gif"}
        ]
    },
    "neck": {
        "거북목 증후군": [
            {"type": "강화", "name": "목 심부 굴곡근 강화 (Chin Tuck & Neck Retraction)", "desc": "턱을 당겨 뒤통수를 벽에 붙이는 동작을 통해, 목의 안정성을 담당하는 가장 중요한 심부 근육을 강화합니다.", "img": "static/exercises/chin_tuck.gif"},
            {"type": "이완", "name": "후두하근 및 상부승모근 이완", "desc": "마사지볼로 목 뒤쪽을 풀어주거나, 손으로 머리를 당겨 목 옆쪽을 스트레칭하여 과긴장된 근육을 이완시킵니다.", "img": "static/exercises/neck_stretch.jpg"}
        ],
        "굽은 등 보상 패턴": [
            {"type": "이완", "name": "가슴 스트레칭 (Pectoral Stretch in Doorway)", "desc": "문틀을 잡고 몸을 앞으로 내밀어, 짧아지고 굽은 어깨와 가슴 근육을 활짝 펴줍니다.", "img": "static/exercises/pec_stretch.jpg"},
            {"type": "강화", "name": "견갑골 안정화 운동 (Wall Angels & Scapular Push-ups)", "desc": "벽에 등을 대고 팔을 움직이거나 네발기기 자세에서 날개뼈를 움직여, 등 뒤쪽의 약화된 근육을 강화하고 날개뼈의 정상적인 움직임을 회복합니다.", "img": "static/exercises/wall_angels.gif"}
        ]
    },
    "knee": {
        "대퇴사두근 우세 패턴": [
            {"type": "강화", "name": "둔근 활성화 및 강화 운동 (Glute Bridge & Single Leg RDL)", "desc": "엉덩이 근육을 먼저 사용하는 법을 다시 인지시키고 강화하여, 무릎의 부담을 근본적으로 줄여줍니다.", "img": "static/exercises/single_leg_rdl.gif"},
            {"type": "이완", "name": "대퇴사두근 폼롤러 마사지 (Quad Foam Rolling)", "desc": "폼롤러를 이용해 과도하게 긴장된 허벅지 앞쪽 근육을 이완시켜 슬개골의 압박을 줄여줍니다.", "img": "static/exercises/quad_roll.gif"}
        ],
        "충격 흡수 부전": [
            {"type": "이완", "name": "발목 가동성 운동 (Ankle Mobility Drill)", "desc": "발목의 배측굴곡(Dorsiflexion) 가동범위를 늘려, 1차 충격 흡수 장치인 발목의 유연성을 개선합니다.", "img": "static/exercises/ankle_mobility.gif"},
            {"type": "강화", "name": "신경근 조절 훈련 (Low-level Plyometrics)", "desc": "가벼운 줄넘기나 제자리 점프 훈련을 통해, 충격을 부드럽게 흡수하는 신경-근육 협응 능력을 향상시킵니다.", "img": "static/exercises/jumping.gif"}
        ],
        "무릎 과신전 (Genu Recurvatum)": [
            {"type": "강화", "name": "햄스트링 강화 운동 (Hamstring Curls & Nordic Hamstring)", "desc": "무릎이 뒤로 꺾이지 않도록 브레이크 역할을 하는 허벅지 뒤 근육을 집중적으로 강화합니다.", "img": "static/exercises/hamstring_curl.gif"},
            {"type": "인지", "name": "고유수용성감각 훈련 (Proprioception Training)", "desc": "한 발로 서서 균형을 잡거나 눈을 감고 걸으며, 무릎의 정상적인 위치를 뇌가 다시 인지하도록 훈련합니다.", "img": "static/exercises/balance.jpg"}
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 전송되지 않았습니다.'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일을 선택해주세요.'})

    if file:
        analysis_data = None
        result_image_local_path = None
        analysis_type = None
        final_result_image_url = None
        public_id = None
        upload_result = None

        try:
            print("Uploading original file to Cloudinary...")
            upload_result = cloudinary.uploader.upload(
                file,
                resource_type = "auto",
                folder = "pose_uploads"
            )
            media_url = upload_result['secure_url']
            print(f"File uploaded to Cloudinary: {media_url}")
            public_id = upload_result.get('public_id')

            print("Starting analysis...")
            if upload_result.get('resource_type') == 'video':
                temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                print(f"Downloading video temporarily to {temp_video_path}")
                response = requests.get(media_url, stream=True)
                response.raise_for_status()
                with open(temp_video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                analysis_data, result_image_local_path, analysis_type = analyzer_module.run_analysis(temp_video_path)
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                    print(f"Temporary video file deleted: {temp_video_path}")
            else:
                analysis_data, result_image_local_path, analysis_type = analyzer_module.run_analysis_from_url(media_url)

            print("Analysis complete.")

            if analysis_data and result_image_local_path and os.path.exists(result_image_local_path):
                print(f"Uploading result image from {result_image_local_path} to Cloudinary...")
                result_public_id = f"result_{public_id}" if public_id else f"result_{os.path.splitext(secure_filename(file.filename))[0]}"

                result_upload = cloudinary.uploader.upload(
                    result_image_local_path,
                    public_id=result_public_id,
                    folder = "pose_results"
                )
                final_result_image_url = result_upload['secure_url']
                print(f"Result image uploaded to Cloudinary: {final_result_image_url}")

                if os.path.exists(result_image_local_path):
                    os.remove(result_image_local_path)
                    print(f"Temporary result file deleted: {result_image_local_path}")

                expert_diagnoses = []
                exercise_prescriptions = []
                for part in ["back", "neck", "knee"]:
                    pattern = analysis_data.get(f"{part}_pattern")
                    if pattern:
                        if pattern in EXPERT_KNOWLEDGE_BASE.get(part, {}):
                            expert_diagnoses.append({
                                "part": part.upper(), "pattern": pattern, "risk": EXPERT_KNOWLEDGE_BASE[part][pattern]
                            })
                        if pattern in EXERCISE_PRESCRIPTION_DB.get(part, {}):
                            prescriptions = EXERCISE_PRESCRIPTION_DB[part][pattern]
                            exercise_prescriptions.append({
                                "part": f"{part.upper()} 개선 솔루션", "exercises": prescriptions
                            })

                return jsonify({
                    'basic_result': analysis_data,
                    'expert_diagnoses': expert_diagnoses,
                    'exercise_prescriptions': exercise_prescriptions,
                    'image_url': final_result_image_url,
                    'analysis_type': analysis_type
                })
            else:
                error_msg = '파일에서 자세를 감지하지 못했습니다.'
                if result_image_local_path and not os.path.exists(result_image_local_path):
                    error_msg += " (결과 이미지 생성 실패)"
                print(error_msg)
                # 분석 실패 시 원본 삭제
                if public_id and upload_result:
                    try:
                        cloudinary.uploader.destroy(public_id, resource_type=upload_result.get('resource_type', 'image'))
                        print(f"Original upload deleted (analysis failure): {public_id}")
                    except Exception as delete_error:
                        print(f"Failed to delete original upload {public_id}: {delete_error}")
                return jsonify({'error': error_msg})

        except cloudinary.exceptions.Error as cloud_error:
             print(f"Cloudinary Error: {cloud_error}")
             return jsonify({'error': f'Cloudinary API 오류: {cloud_error}'}), 500 # Cloudinary 오류 명시
        except Exception as e:
            import traceback
            print(f"An error occurred during analysis: {e}")
            traceback.print_exc() # 상세 오류 스택 출력

            # 오류 시 임시 파일 삭제
            if result_image_local_path and os.path.exists(result_image_local_path):
                try: os.remove(result_image_local_path); print(f"Cleaned up temp result file: {result_image_local_path}")
                except Exception as del_err: print(f"Could not delete temp result file {result_image_local_path}: {del_err}")
            # 동영상 임시 파일도 삭제 시도 (만약 존재한다면)
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                 try: os.remove(temp_video_path); print(f"Cleaned up temp video file: {temp_video_path}")
                 except Exception as del_err: print(f"Could not delete temp video file {temp_video_path}: {del_err}")

            # 오류 시 원본 삭제 (주의: public_id가 정의되었는지 확인 필요)
            if public_id and upload_result:
                try:
                    cloudinary.uploader.destroy(public_id, resource_type=upload_result.get('resource_type', 'image'))
                    print(f"Original upload deleted (error): {public_id}")
                except Exception as delete_error:
                    print(f"Failed to delete original upload {public_id} after error: {delete_error}")

            return jsonify({'error': f'서버 내부 오류 발생: {e}'}), 500

    return jsonify({'error': '파일 처리 중 알 수 없는 오류 발생'})


if __name__ == '__main__':
    # 임시 폴더가 로컬 테스트 시에만 생성되도록 보장
    TEMP_FOLDER = app.config.get('UPLOAD_FOLDER', 'static/uploads')
    if not os.path.exists(TEMP_FOLDER):
        os.makedirs(TEMP_FOLDER)

    # Cloud Run이 지정하는 $PORT를 사용하고, 없으면 5001 사용
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=True)