from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import base64
import numpy as np
import scipy.signal
import scipy.fft
import soundfile as sf
import matplotlib
matplotlib.use('Agg') # 서버 에러 방지용 필수 설정
import matplotlib.pyplot as plt
import librosa
import librosa.display
import gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# BORA LABS DSP KERNEL (ORIGINAL LOGIC PORTED)
# ==============================================================================

class MasteringDSP:
    @staticmethod
    def load_audio(file_obj):
        try:
            # [중요] 원본 로직 복원: Intro Skip Logic
            # 파일 객체로부터 길이를 먼저 알아내야 합니다.
            with sf.SoundFile(file_obj) as f:
                duration = len(f) / f.samplerate
                sr_native = f.samplerate
            
            # 파일 포인터 초기화 (길이 쟀으니 다시 처음으로)
            file_obj.seek(0)

            # 원본 로직: 40초 이상이면 앞부분 30초 건너뛰고 하이라이트 분석
            start_time = 30.0 if duration > 40.0 else 0.0
            
            # Librosa 로딩 (offset 적용)
            y, sr = librosa.load(file_obj, sr=44100, mono=False, offset=start_time, duration=180)
            
            if y.ndim == 1: y = np.stack((y, y))
            return y, sr
        except Exception as e:
            # MP3 등 SoundFile이 못 읽는 경우를 대비한 안전장치
            print(f"Direct load fallback due to: {e}")
            file_obj.seek(0)
            y, sr = librosa.load(file_obj, sr=44100, mono=False, duration=180)
            if y.ndim == 1: y = np.stack((y, y))
            return y, sr

    @staticmethod
    def k_weighting_filter(y, sr):
        # 원본 필터 계수 유지
        b1, a1 = scipy.signal.iirfilter(1, 1500/(sr/2), btype='high', ftype='butter') 
        y_h = scipy.signal.lfilter(b1, a1, y)
        b2, a2 = scipy.signal.butter(1, 38/(sr/2), btype='high')
        y_k = scipy.signal.lfilter(b2, a2, y_h)
        return y_k

    @staticmethod
    def analyze_loudness_dynamics(y_stereo, sr):
        # 원본 LUFS/PLR 계산 로직
        y_resampled = scipy.signal.resample(y_stereo, int(y_stereo.shape[1] * 2), axis=1)
        true_peak = np.max(np.abs(y_resampled))
        true_peak_db = 20 * np.log10(true_peak + 1e-9)
        
        y_k_L = MasteringDSP.k_weighting_filter(y_stereo[0], sr)
        y_k_R = MasteringDSP.k_weighting_filter(y_stereo[1], sr)
        mean_power = (np.mean(y_k_L**2) + np.mean(y_k_R**2)) / 2.0
        lufs_approx = -0.691 + 10 * np.log10(mean_power + 1e-9)
        
        plr = true_peak_db - lufs_approx
        
        return round(float(lufs_approx), 1), round(float(true_peak_db), 1), round(float(plr), 1)

    @staticmethod
    def analyze_stereo_image(y_stereo):
        if y_stereo.shape[0] < 2: return 1.0, 0.0
        L, R = y_stereo[0], y_stereo[1]
        
        dot_prod = np.mean((L - np.mean(L)) * (R - np.mean(R)))
        std_prod = np.std(L) * np.std(R) + 1e-9
        correlation = dot_prod / std_prod
        
        side = (L - R) * 0.5
        mid = (L + R) * 0.5
        side_energy = np.sum(side**2)
        mid_energy = np.sum(mid**2) + 1e-9
        width_percent = (side_energy / (mid_energy + side_energy)) * 100
        
        return round(float(correlation), 2), round(float(width_percent), 1)

    @staticmethod
    def analyze_frequency_spectrum(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        spec = np.abs(librosa.stft(y_mono, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        
        bands = {
            "SUB": (20, 60), "BASS": (60, 250),
            "LOW_MID": (250, 2000), "HIGH_MID": (2000, 6000), "AIR": (6000, 20000)
        }
        
        energy_dist = {}
        total_energy = np.sum(spec) + 1e-9
        
        for name, (f_min, f_max) in bands.items():
            mask = (freqs >= f_min) & (freqs < f_max)
            band_energy = np.sum(spec[mask, :])
            energy_dist[name] = float((band_energy / total_energy) * 100)
            
        return energy_dist

    @staticmethod
    def analyze_musical_key(y_stereo, sr):
        # 원본 키 감지 로직 (CENS Chroma)
        try:
            y_mono = librosa.to_mono(y_stereo)
            y_harm, _ = librosa.effects.hpss(y_mono, margin=3.0)
            chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_mean /= np.linalg.norm(chroma_mean) # 정규화 추가 (원본 동일)
            
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            maj_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            min_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            maj_p /= np.linalg.norm(maj_p)
            min_p /= np.linalg.norm(min_p)
            
            best_score, best_key = -1, ""
            for i in range(12):
                if np.dot(chroma_mean, np.roll(maj_p, i)) > best_score:
                    best_score = np.dot(chroma_mean, np.roll(maj_p, i))
                    best_key = f"{keys[i]} Maj"
                if np.dot(chroma_mean, np.roll(min_p, i)) > best_score:
                    best_score = np.dot(chroma_mean, np.roll(min_p, i))
                    best_key = f"{keys[i]} Min"
                    
            return best_key
        except:
            return "Unknown"

    @staticmethod
    def generate_spectrogram_base64(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        if len(y_mono) > sr * 30: y_vis = y_mono[:sr*30] 
        else: y_vis = y_mono
            
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_vis)), ref=np.max)
        
        # Bora Labs 스타일에 맞춰 배경은 투명/블랙 처리
        fig = plt.figure(figsize=(10, 3), facecolor='none') 
        ax = plt.axes()
        ax.set_facecolor('none')
        
        # 원본과 동일하게 cmap='ocean' 사용
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='ocean', ax=ax)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

class AIAdviser:
    @staticmethod
    def generate_report(data):
        # 원본 텍스트 & 로직 100% 동일
        advice_list = []
        if data['lufs'] > -7:
            advice_list.append({"status": "WARN", "title": "High Loudness Level", "msg": "음압이 매우 높습니다(-7 LUFS 초과). 클럽/CD 마스터링 의도가 아니라면 다이내믹 레인지 손실을 체크하세요."})
        else:
            advice_list.append({"status": "PASS", "title": "Safe Loudness Range", "msg": "다이내믹 레인지가 보존된 안전한 음압 범위 내에 있습니다."})

        if data['corr'] < 0.2:
            advice_list.append({"status": "CRIT", "title": "Phase Cancellation Risk", "msg": "위상 상관도가 낮습니다. 모노 환경(스마트폰 등)에서 소리가 사라질 수 있습니다."})
        elif data['corr'] < 0.6:
            advice_list.append({"status": "INFO", "title": "Wide Stereo Image", "msg": "넓은 스테레오 이미지를 가지고 있습니다."})
        else:
            advice_list.append({"status": "PASS", "title": "Solid Phase Coherence", "msg": "위상 호환성이 좋으며 단단한 믹스 밸런스를 유지하고 있습니다."})

        sub_energy = data['freq']['SUB']
        air_energy = data['freq']['AIR']
        
        if sub_energy > 25:
            advice_list.append({"status": "WARN", "title": "Excessive Low-End", "msg": f"SUB 대역 에너지가 {int(sub_energy)}%로 높습니다. 불필요한 초저역 부밍(Booming)이 없는지 확인하세요."})
        elif sub_energy < 5:
            advice_list.append({"status": "WARN", "title": "Lack of Low-End", "msg": "저음역 에너지가 부족합니다. 믹스가 전체적으로 가볍게 들릴 수 있습니다."})
            
        if air_energy > 15:
            advice_list.append({"status": "WARN", "title": "Bright High-End", "msg": f"AIR 대역 에너지가 {int(air_energy)}%로 높습니다. 보컬의 치찰음이나 심벌의 자극적인 대역을 디에서로 제어하세요."})

        return advice_list

    @staticmethod
    def generate_matrix(data):
        # 원본 매트릭스 로직
        if data['plr'] < 8: d_val = "COMPRESSED"
        elif data['plr'] > 14: d_val = "DYNAMIC"
        else: d_val = "BALANCED"

        sub = data['freq']['SUB']
        air = data['freq']['AIR']
        if sub > 25: t_val = "BOOMY / DEEP"
        elif air > 15: t_val = "BRIGHT / AIRY"
        elif sub < 10 and air < 8: t_val = "MID-FOCUSED"
        else: t_val = "NEUTRAL"

        corr = data['corr']
        if corr < 0.3: i_val = "PHASE ISSUE"
        elif corr < 0.6: i_val = "WIDE"
        else: i_val = "CENTERED"
        
        lufs = data['lufs']
        if lufs > -9: l_val = "LOUD / CD"
        elif lufs < -16: l_val = "GENTLE"
        else: l_val = "STREAMING"
        
        return [d_val, t_val, i_val, l_val]

# ==============================================================================
# API ENDPOINT
# ==============================================================================

@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    try:
        # 파일을 메모리에 읽기
        contents = await file.read()
        virtual_file = io.BytesIO(contents)
        
        # 1. 오디오 로드 (Intro Skip 로직 적용됨)
        y, sr = MasteringDSP.load_audio(virtual_file)
        
        # 2. 분석 실행 (원본 로직)
        lufs, tp, plr = MasteringDSP.analyze_loudness_dynamics(y, sr)
        corr, width = MasteringDSP.analyze_stereo_image(y)
        freq_dist = MasteringDSP.analyze_frequency_spectrum(y, sr)
        key = MasteringDSP.analyze_musical_key(y, sr)
        spec_img = MasteringDSP.generate_spectrogram_base64(y, sr)
        
        # 3. 데이터 패키징
        analysis_data = {
            "lufs": lufs, "true_peak": tp, "plr": plr,
            "corr": corr, "width": width,
            "freq": freq_dist,
            "key": key,
            "spec_img": f"data:image/png;base64,{spec_img}"
        }
        
        # 4. 리포트 & 매트릭스 생성
        report = AIAdviser.generate_report(analysis_data)
        matrix = AIAdviser.generate_matrix(analysis_data) # [D, T, I, L] 리스트 반환

        del y, contents
        gc.collect()

        # 프론트엔드로 보낼 최종 JSON
        return {
            "status": "success",
            "data": {
                **analysis_data, 
                "report": report,
                "matrix": matrix # 프론트엔드에서 4가지 요약(Dynamics 등) 표시용
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}