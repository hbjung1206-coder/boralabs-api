from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import gc  # 🗑️ 쓰레기 청소부 (메모리 관리용)

app = FastAPI()

# 1. 보안 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 헬스 체크
@app.get("/")
def read_root():
    return {"status": "online", "message": "Bora Labs API is optimized & running!"}

# 3. 오디오 분석
@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # 파일 저장
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # =========================================================
        # 🚨 [중요] 메모리 폭발 방지 패치
        # 1. duration=60: 60초까지만 분석 (무료 서버 한계 극복)
        # 2. sr=22050: 샘플링 레이트를 표준값으로 고정 (데이터 크기 절반 감소)
        # 3. mono=False: 스테레오 분석을 위해 유지
        # =========================================================
        y, sr = librosa.load(temp_filename, sr=22050, mono=False, duration=60.0)
        
        # 데이터가 Mono(1채널)로 들어온 경우 형상 맞추기
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=0) # (N,) -> (1, N)
            
        # --- 분석 시작 ---

        # 1. LUFS (Stereo 평균)
        # y의 shape이 (채널, 샘플)임.
        rms = np.sqrt(np.mean(y**2))
        lufs = float(20 * np.log10(rms)) if rms > 0 else -70.0
        
        # 2. True Peak
        true_peak = np.max(np.abs(y))
        true_peak_db = float(20 * np.log10(true_peak)) if true_peak > 0 else -70.0
        
        # 3. Stereo Width
        if y.shape[0] > 1: # 스테레오라면
            y_left = y[0]
            y_right = y[1]
            # 길이가 짧으면 상관계수 계산이 위험할 수 있으므로 예외처리
            if len(y_left) > 100:
                corr_matrix = np.corrcoef(y_left, y_right)
                corr = float(corr_matrix[0, 1])
            else:
                corr = 1.0
            width = (1 - corr) * 100
        else: # 모노라면
            corr = 1.0
            width = 0.0

        # 4. Key Detection (메모리 절약을 위해 모노로 변환해서 분석)
        y_mono = librosa.to_mono(y)
        chroma = librosa.feature.chroma_stft(y=y_mono, sr=sr)
        key_idx = np.argmax(np.sum(chroma, axis=1))
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        detected_key = keys[key_idx]

        # 5. Frequency Balance (메모리 절약을 위해 FFT 크기 조절)
        S = np.abs(librosa.stft(y_mono, n_fft=2048, hop_length=1024)) # hop_length를 늘려 데이터 감소
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        energy = np.sum(S, axis=1)
        
        total_e = np.sum(energy)
        if total_e == 0: total_e = 1
        
        def get_energy_pct(min_f, max_f, boost=1.0):
            indices = np.where((freq_bins >= min_f) & (freq_bins < max_f))[0]
            if len(indices) == 0: return 0
            val = (np.sum(energy[indices]) / total_e) * 100 * boost
            return float(val)

        freq_data = {
            "SUB": get_energy_pct(20, 60, 3.0),
            "BASS": get_energy_pct(60, 250, 2.0),
            "LOW_MID": get_energy_pct(250, 2000, 1.0),
            "HIGH_MID": get_energy_pct(2000, 6000, 1.0),
            "AIR": get_energy_pct(6000, 20000, 2.5)
        }

        # 6. Spectrogram Image (이미지 사이즈 축소)
        plt.figure(figsize=(8, 3)) # 크기 줄임
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), sr=sr, y_axis='log', x_axis='time', cmap='magma')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=100) # DPI 낮춤
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # 메모리 정리 (중요!)
        del y, y_mono, S, chroma
        gc.collect()

        # 7. Report Logic
        report = []
        if lufs < -14:
            report.append({"status": "WARN", "title": "LOW LOUDNESS", "msg": "음압이 낮습니다 (-14 LUFS 이하)."})
        elif lufs > -6:
             report.append({"status": "CRIT", "title": "OVER COMPRESSED", "msg": "음압이 너무 높습니다. 다이내믹이 죽었어요."})
        else:
            report.append({"status": "PASS", "title": "LOUDNESS OK", "msg": "적절한 음압입니다."})

        if true_peak_db > -1.0:
             report.append({"status": "WARN", "title": "PEAK DANGER", "msg": "피크가 뜹니다 (-1dB 초과)."})
        
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
                "matrix": ["Dynamic", "Warm", "Wide", "Club"]
            }
        }

    except Exception as e:
        print(f"Error: {str(e)}") # 로그 확인용
        return {"status": "error", "message": "File too big or format error"}
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect() # 마지막 청소