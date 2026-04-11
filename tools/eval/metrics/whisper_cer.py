"""Whisper CER (Character Error Rate) evaluation for voice conversion quality."""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# Whisper transcription parameters tuned for singing voice
WHISPER_SINGING_PARAMS = {
    "beam_size": 1,
    "best_of": 5,
    "temperature": 0,
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.5,
    "compression_ratio_threshold": 3.5,
}


def normalize_japanese(text: str) -> str:
    """Normalize Japanese text for CER comparison."""
    # NFKC unicode normalization
    text = unicodedata.normalize("NFKC", text)

    import jaconv

    # Full-width alphanumerics to half-width (keep kana as-is)
    text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
    # Katakana to hiragana for unified comparison
    text = jaconv.kata2hira(text)
    # Remove long vowel mark
    text = text.replace("\u30fc", "")
    # Remove punctuation and symbols
    text = re.sub(r"[、。！？\s\.,!?\-\(\)（）「」『』\[\]【】…·♪]", "", text)
    # Remove any remaining whitespace
    text = text.strip()

    return text


def _load_whisper_model(model_name: str, device: str):
    """Load Whisper model with device fallback."""
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisperがインストールされていません。"
            " `pip install openai-whisper` または `uv add openai-whisper` でインストールしてください。"
        )

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDAが利用できないため、CPUにフォールバックします")
        device = "cpu"

    logger.info("Whisperモデル '%s' をロード中 (device=%s)", model_name, device)
    model = whisper.load_model(model_name, device=device)
    return model


def _transcribe(model, audio_path: str, language: str) -> str:
    """Transcribe audio file using Whisper.

    Returns the raw transcription text (before normalization).
    """
    import whisper

    audio = whisper.load_audio(audio_path)
    result = model.transcribe(audio, language=language, **WHISPER_SINGING_PARAMS)
    return result["text"]


def compute_ref_cer(
    ref_path: str,
    ref_text: str,
    model_name: str = "large-v3",
    language: str = "ja",
    device: str = "cuda",
) -> dict:
    """Compute CER of reference audio alone by comparing Whisper transcription against ground-truth.

    This measures how well Whisper can recognize the reference audio, which serves as
    a baseline for interpreting voice conversion CER results. For singing voice, Whisper
    recognition accuracy is often low (CER 20-30%), so knowing the reference audio's
    CER is essential for fair evaluation.

    Args:
        ref_path: Path to the reference audio file.
        ref_text: Ground-truth text for the reference audio.
        model_name: Whisper model size (tiny, base, small, medium, large, large-v3).
        language: Language code for transcription.
        device: Compute device ("cuda" or "cpu").

    Returns:
        Dict with CER value, unit, and details including transcribed and reference texts.
    """
    try:
        import jiwer
    except ImportError:
        raise ImportError(
            "jiwerがインストールされていません。"
            " `pip install jiwer` または `uv add jiwer` でインストールしてください。"
        )

    model = _load_whisper_model(model_name, device)

    logger.info("参照音声を転写中 (ref CER計算): %s", ref_path)
    transcribed_raw = _transcribe(model, ref_path, language)
    transcribed_normalized = normalize_japanese(transcribed_raw)
    ref_text_normalized = normalize_japanese(ref_text)

    logger.info("参照テキスト (正規化後): %s", ref_text_normalized)
    logger.info("転写結果 (正規化後): %s", transcribed_normalized)

    if not ref_text_normalized:
        logger.warning("参照テキストが空のためCER計算不能")
        cer = 0.0 if not transcribed_normalized else 1.0
    else:
        cer = jiwer.cer(ref_text_normalized, transcribed_normalized)

    return {
        "value": cer,
        "unit": "ratio",
        "details": {
            "ref_text": ref_text_normalized,
            "ref_text_raw": ref_text,
            "transcribed_text": transcribed_normalized,
            "transcribed_text_raw": transcribed_raw,
            "model": model_name,
        },
    }


def compute_whisper_cer(
    ref_path: str,
    conv_path: str,
    ref_text: str | None = None,
    model_name: str = "large-v3",
    language: str = "ja",
    device: str = "cuda",
) -> dict:
    """Compute Character Error Rate between reference and converted audio using Whisper.

    Two modes of operation depending on whether ``ref_text`` is provided:

    **Absolute mode** (``ref_text`` provided, ``metric_type="absolute"``):
        CER is computed between the provided ground-truth text and the Whisper
        transcription of the converted audio. ``ref_audio_cer`` is also computed
        (ground-truth vs. Whisper transcription of the reference audio) and
        ``delta_cer`` represents the degradation: ``conv_cer - ref_audio_cer``.

    **Relative mode** (``ref_text=None``, ``metric_type="relative"``):
        Both reference and converted audio are transcribed by Whisper, and
        CER is computed between the two transcriptions. In this mode,
        ``delta_cer`` equals ``cer_value`` because the reference transcription
        itself is the baseline (i.e., ref_audio_cer is implicitly 0). This
        measures how much the voice conversion changed the recognized text
        content, but cannot distinguish Whisper recognition errors from actual
        conversion artifacts.

    Args:
        ref_path: Path to the reference audio file.
        conv_path: Path to the converted audio file.
        ref_text: Optional ground-truth text. If None, Whisper transcribes the reference audio.
        model_name: Whisper model size (tiny, base, small, medium, large, large-v3).
        language: Language code for transcription.
        device: Compute device ("cuda" or "cpu").

    Returns:
        Dict with CER value, unit, and details including transcribed texts.
        Details include:
            - ref_text: normalized reference text
            - ref_text_raw: raw reference text (before normalization)
            - conv_text: normalized converted audio transcription
            - conv_text_raw: raw converted audio transcription (before normalization)
            - ref_source: "provided" or "whisper"
            - ref_audio_cer: CER of reference audio vs ground-truth (None when ref_text not provided)
            - delta_cer: degradation from reference audio
            - metric_type: "absolute" (ref_text provided) or "relative" (ref_text not provided)
    """
    try:
        import jiwer
    except ImportError:
        raise ImportError(
            "jiwerがインストールされていません。"
            " `pip install jiwer` または `uv add jiwer` でインストールしてください。"
        )

    model = _load_whisper_model(model_name, device)

    # Determine reference text
    ref_audio_cer = None
    if ref_text is not None:
        ref_source = "provided"
        metric_type = "absolute"
        ref_text_raw = ref_text
        ref_text_normalized = normalize_japanese(ref_text)
    else:
        ref_source = "whisper"
        metric_type = "relative"
        logger.info("参照音声を転写中: %s", ref_path)
        ref_text_raw = _transcribe(model, ref_path, language)
        ref_text_normalized = normalize_japanese(ref_text_raw)
        logger.info("参照音声の転写結果: %s", ref_text_normalized)

    # Transcribe converted audio
    logger.info("変換音声を転写中: %s", conv_path)
    conv_text_raw = _transcribe(model, conv_path, language)
    conv_text_normalized = normalize_japanese(conv_text_raw)
    logger.info("変換音声の転写結果: %s", conv_text_normalized)

    # Compute CER
    if not ref_text_normalized:
        logger.warning("参照テキストが空のためCER計算不能")
        cer_value = 0.0 if not conv_text_normalized else 1.0
    else:
        cer_value = jiwer.cer(ref_text_normalized, conv_text_normalized)

    # Compute ref_audio_cer when ref_text was provided (compare provided text vs Whisper transcription of ref audio)
    if ref_text is not None:
        logger.info("参照音声のWhisper転写CERを計算中")
        ref_audio_text_raw = _transcribe(model, ref_path, language)
        ref_audio_text_normalized = normalize_japanese(ref_audio_text_raw)
        if ref_text_normalized:
            ref_audio_cer = jiwer.cer(ref_text_normalized, ref_audio_text_normalized)
        else:
            ref_audio_cer = 0.0

    # delta_cer: degradation from reference audio
    if ref_text is not None and ref_audio_cer is not None:
        delta_cer = cer_value - ref_audio_cer
    else:
        # Relative mode: delta_cer equals cer_value because the reference
        # transcription is the baseline itself (ref_audio_cer is implicitly 0).
        delta_cer = cer_value

    return {
        "value": cer_value,
        "unit": "ratio",
        "details": {
            "ref_text": ref_text_normalized,
            "ref_text_raw": ref_text_raw,
            "conv_text": conv_text_normalized,
            "conv_text_raw": conv_text_raw,
            "ref_source": ref_source,
            "ref_audio_cer": ref_audio_cer,
            "delta_cer": delta_cer,
            "metric_type": metric_type,
        },
    }
