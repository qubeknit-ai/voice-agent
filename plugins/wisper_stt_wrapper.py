from __future__ import annotations
import asyncio
from livekit.agents import stt
from livekit import rtc
from faster_whisper import WhisperModel
import numpy as np


class FasterWhisperSTT(stt.STT):
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: str | None = None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        # buffer is a single AudioFrame
        audio = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()

        def transcribe():
            segments, _ = self._model.transcribe(
                audio, beam_size=5, language=language or "en"
            )
            return " ".join(s.text for s in segments).strip()

        text = await loop.run_in_executor(None, transcribe)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "en")],
        )

    def stream(self, *, language=None, conn_options=None):
        raise NotImplementedError("Use VAD-based streaming via AgentSession instead")