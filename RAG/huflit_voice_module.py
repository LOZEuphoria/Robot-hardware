import logging
import os
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from piper import PiperVoice


logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STT_MODEL_NAME = os.getenv("WHISPER_MODEL", "faster-whisper-large-v3-turbo")
STT_MODEL_DIR = BASE_DIR / "models" / STT_MODEL_NAME
TTS_MODEL_PATH = BASE_DIR / "models" / "vi_VN-vais1000-medium.onnx"
TTS_CONFIG_PATH = BASE_DIR / "models" / "vi_VN-vais1000-medium.onnx.json"


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {description}: {path}")


_require_path(STT_MODEL_DIR, "model faster-whisper")
_require_path(TTS_MODEL_PATH, "model Piper")
_require_path(TTS_CONFIG_PATH, "cấu hình Piper")

logging.info("Đang nạp model STT Whisper %s từ ổ đĩa...", STT_MODEL_NAME)
stt_model = WhisperModel(
    str(STT_MODEL_DIR),
    device="cpu",
    compute_type="int8",
    cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "8")),
    num_workers=1,
    local_files_only=True,
)

logging.info("Đang nạp model TTS Piper tiếng Việt từ ổ đĩa...")
tts_model = PiperVoice.load(
    str(TTS_MODEL_PATH),
    config_path=str(TTS_CONFIG_PATH),
)


def transcribe_audio(audio_path: str) -> str:
    """Chuyển file âm thanh thành văn bản tiếng Việt hoàn toàn offline."""
    if not audio_path:
        return ""

    path = Path(audio_path)
    if not path.is_file():
        logging.error("Không tìm thấy file âm thanh: %s", path)
        return ""

    logging.info("Đang nhận diện giọng nói từ: %s", path)
    segments, info = stt_model.transcribe(
        str(path),
        beam_size=5,
        language="vi",
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=(
            "Đây là câu hỏi của sinh viên dành cho trợ lý HUFLIT về học phí, "
            "học bổng, tuyển sinh, lịch thi, quy chế và biểu mẫu."
        ),
        hotwords=(
            "xin thông tin HUFLIT học phí học bổng tuyển sinh lịch thi thời khóa biểu "
            "bảo hiểm y tế biểu mẫu phòng đào tạo"
        ),
        vad_filter=True,
        vad_parameters={
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 500,
        },
    )
    result = " ".join(segment.text.strip() for segment in segments).strip()
    logging.info(
        "Whisper language=%s probability=%.3f duration=%.2fs after_vad=%.2fs",
        info.language,
        info.language_probability,
        info.duration,
        info.duration_after_vad,
    )
    logging.info("Nhận diện được: %s", result)
    return result


def prepare_text_for_speech(text: str) -> str:
    """Turn Markdown/API output into clean Vietnamese prose for Piper."""
    result = str(text)
    result = re.sub(
        r"(?:\bngày\s+)?(\d{1,2})/(\d{1,2})/(\d{4})\b",
        r"ngày \1 tháng \2 năm \3",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", result)
    result = re.sub(r"https?://\S+|www\.\S+", " ", result)
    result = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", result)
    result = re.sub(r"(?m)^\s*[-*+•]\s+", ". ", result)
    result = re.sub(r"(?m)^\s*\d+[.)]\s+", ". ", result)
    result = result.replace("&", " và ").replace("%", " phần trăm ")
    result = re.sub(r"[*_`~<>|{}\[\]]", "", result)
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\s*\n+\s*", ". ", result)
    result = re.sub(r"(?:\.\s*){2,}", ". ", result)
    return result.strip(" .,")


def generate_speech(text: str) -> Optional[str]:
    """Chuyển văn bản thành file WAV tiếng Việt hoàn toàn offline."""
    if not text or not text.strip():
        return None

    speech_text = prepare_text_for_speech(text)
    if not speech_text:
        return None

    output_path = (
        Path(tempfile.gettempdir()) / f"huflit_response_{uuid.uuid4()}.wav"
    )

    try:
        logging.info("Đang tạo giọng nói tiếng Việt bằng Piper local...")
        with wave.open(str(output_path), "wb") as wav_file:
            tts_model.synthesize_wav(speech_text, wav_file)
        return str(output_path)
    except Exception:
        logging.exception("Không thể tạo giọng nói bằng Piper")
        output_path.unlink(missing_ok=True)
        return None


if __name__ == "__main__":
    sample = "Chào mừng bạn đến với Trường Đại học Ngoại ngữ - Tin học Thành phố Hồ Chí Minh."
    audio_file = generate_speech(sample)
    print(f"Đã tạo file âm thanh: {audio_file}")
