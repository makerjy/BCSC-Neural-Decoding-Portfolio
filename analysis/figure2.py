import mne
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 파일 경로를 Path 객체로 정의
set_file_path = Path("Analysis - Speech Decoding Dataset/P01/Control/P01_10_remove_bad_components_2_c.set")

# 경로 확인용 디버깅
print(f"🔍 현재 실행 경로(CWD): {Path.cwd()}")
print(f"📁 찾고자 하는 절대 경로: {set_file_path.resolve()}")

# 파일 존재 여부 검사
if not set_file_path.exists():
    raise FileNotFoundError(
        f"\n❌ 파일을 찾을 수 없습니다!\n"
        f"현재 위치({Path.cwd()}) 기준으로 해당 파일이 없습니다.\n"
        f"경로 스펠링이나 띄어쓰기를 확인해 주세요."
    )

# 2. .set 파일 로드 (Raw 또는 Epochs 자동 감지 및 예외 처리)
try:
    data = mne.io.read_raw_eeglab(set_file_path, preload=False)
    print("ℹ️ Raw 형태의 EEGLAB 파일로 로드되었습니다.")
except (TypeError, ValueError):
    # .set 파일 내에 에포크(Epochs) 트라이얼 정보가 들어있는 경우
    data = mne.read_epochs_eeglab(set_file_path)
    print("ℹ️ Epochs 형태의 EEGLAB 파일로 로드되었습니다.")

# 3. 데이터에서 채널 이름 추출
raw_ch_names = data.ch_names
print(f"📋 불러온 원본 채널 수: {len(raw_ch_names)}개")

# 4. 불필요한 노이즈/보조 채널 제거
bad_channels = [
    'EOG', 'M1', 'M2',
    'BIP1', 'BIP2', 'BIP3', 'BIP4', 'BIP5', 'BIP6', 'BIP7', 'BIP8',
    'BIP9', 'BIP10', 'BIP11', 'BIP12', 'BIP13', 'BIP14', 'BIP15', 'BIP16',
    'BIP17', 'BIP18', 'BIP19', 'BIP20', 'BIP21', 'BIP22', 'BIP23', 'BIP24'
]

channels_to_drop = [ch for ch in bad_channels if ch in raw_ch_names]
if channels_to_drop:
    data.drop_channels(channels_to_drop)
    print(f"🗑️ 제거된 불필요 채널 ({len(channels_to_drop)}개): {channels_to_drop}")

eeg_ch_names = data.ch_names
print(f"✅ 최종 추출된 EEG 전극 수: {len(eeg_ch_names)}개")
print(f"📌 채널 목록:\n{eeg_ch_names}\n")

# 5. 표준 몽타주 매칭 및 기준 채널 맵 시각화
montage = mne.channels.make_standard_montage('standard_1020')
data.set_montage(montage, match_case=False, on_missing='ignore')

fig, ax = plt.subplots(figsize=(8, 8))
mne.viz.plot_sensors(
    data.info,
    kind='topomap',
    show_names=True,
    ch_type='eeg',
    axes=ax,
    sphere=(0.0, 0.022, 0.027, 0.105)
)
plt.title(f"Baseline Channel Map ({len(eeg_ch_names)} Channels)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()