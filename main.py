from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = FastAPI()

# ==========================================
# 1. 보안 설정 (CORS) - 누구나 접속 허용
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 주소 허용 (Vercel 포함)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. 헬스 체크 (Render가 확인하는 곳) - 이게 없어서 404가 떴던 겁니다!
# ==========================================
@app.get("/")
def read_root():
    return {"status": "online", "message": "Bora Labs API is running smoothly!"}

# ==========================================
# 3. 오디오 분석 로직 (기존 기능)
# ==========================================
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # 파일 저장
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Librosa 로딩
        y, sr = librosa.load(temp_filename, sr=None)
        
        # 1. LUFS (Loudness) - 약식 계산 (RMS 기반)
        rms = np.sqrt(np.mean(y**2))
        lufs = 20 * np.log10(rms) if rms > 0 else -70
        
        # 2. True Peak
        true_peak = np.max(np.abs(y))
        true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -70
        
        # 3. Stereo Correlation (Mono일 경우 처리)
        if len(y.shape) > 1:
            # 스테레오일 경우
            y_left = y[0]
            y_right = y[1]
            corr = np.corrcoef(y_left, y_right)[0, 1]
            width = (1 - corr) * 100 # 대략적인 Width 계산
        else:
            # 모노일 경우
            corr = 1.0
            width = 0
            
        # 4. Key Detection (Chroma)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_idx]

        # 5. Frequency Balance (5 Bands)
        # 스펙트로그램 에너지 계산
        S = np.abs(librosa.stft(y))
        
        # 주파수 대역별 에너지 합계 (간단화)
        # Sub: 0-60, Bass: 60-250, LowMid: 250-2k, HighMid: 2k-6k, Air: 6k+
        freq_bins = librosa.fft_frequencies(sr=sr)
        energy = np.sum(S, axis=1)
        
        def get_energy(min_f, max_f):
            indices = np.where((freq_bins >= min_f) & (freq_bins < max_f))[0]
            if len(indices) == 0: return 0
            return np.sum(energy[indices])

        total_e = np.sum(energy)
        if total_e == 0: total_e = 1
        
        freq_data = {
            "SUB": (get_energy(20, 60) / total_e) * 100 * 5,   # 보정값
            "BASS": (get_energy(60, 250) / total_e) * 100 * 3,
            "LOW_MID": (get_energy(250, 2000) / total_e) * 100,
            "HIGH_MID": (get_energy(2000, 6000) / total_e) * 100,
            "AIR": (get_energy(6000, 20000) / total_e) * 100 * 2
        }

        # 6. Spectrogram Image 생성
        plt.figure(figsize=(10, 4))
        # 흑백(gray) 컬러맵에 대비를 높여서 사이버펑크 느낌 주기
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time', cmap='magma')
        plt.axis('off') # 축 제거
        plt.tight_layout(pad=0)
        
        # 이미지 메모리에 저장
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 7. AI Report Logic (간단한 규칙 기반)
        report = []
        
        # LOUDNESS Check
        if lufs < -14:
            report.append({"status": "WARN", "title": "LOW LOUDNESS", "msg": "음압이 너무 낮습니다 (-14 LUFS 이하). 스트리밍 서비스에서 소리가 작게 들릴 수 있습니다."})
        elif lufs > -6:
             report.append({"status": "CRIT", "title": "OVER COMPRESSED", "msg": "음압이 너무 높습니다. 다이내믹 레인지가 손실되었을 가능성이 큽니다."})
        else:
            report.append({"status": "PASS", "title": "LOUDNESS OK", "msg": "스트리밍에 적합한 음압 레벨입니다."})

        # PEAK Check
        if true_peak_db > -1.0:
             report.append({"status": "WARN", "title": "PEAK DANGER", "msg": "트루 피크가 -1dB를 넘었습니다. 인코딩 시 클리핑이 발생할 수 있습니다."})
        
        # WIDTH Check
        if width < 30:
             report.append({"status": "WARN", "title": "NARROW STEREO", "msg": "스테레오 이미지가 좁습니다. 좌우 패닝이나 위상 처리가 필요할 수 있습니다."})

        return {
            "status": "success",
            "data": {
                "lufs": round(lufs, 2),
                "true_peak": round(true_peak_db, 2),
                "plr": round(true_peak_db - lufs, 2),
                "corr": round(corr, 2),
                "width": round(width, 1),
                "key": detected_key,
                "freq": freq_data,
                "spec_img": f"data:image/png;base64,{img_base64}",
                "report": report,
                "matrix": ["Dynamic", "Warm", "Wide", "Club"] # 임시 더미 데이터
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)