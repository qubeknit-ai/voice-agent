from __future__ import annotations
import asyncio
import re
from livekit.agents import tts, utils
from kokoro_onnx import Kokoro
import numpy as np

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


def split_sentences(text: str) -> list[str]:
    # split on sentence boundaries, keep chunks meaningful
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


class KokoroTTS(tts.TTS):
    def __init__(self, voice="af_heart", speed=1.0):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._kokoro = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
        self._voice = voice
        self._speed = speed

    def synthesize(self, text: str, *, conn_options=None) -> tts.ChunkedStream:
        return KokoroStream(tts_instance=self, text=text, conn_options=conn_options)


class KokoroStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: KokoroTTS, text: str, conn_options=None):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        loop = asyncio.get_event_loop()
        sentences = split_sentences(self._input_text)

        # initialize once before first sentence
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        for sentence in sentences:
            def _synth(s=sentence):
                samples, _ = self._tts._kokoro.create(
                    s,
                    voice=self._tts._voice,
                    speed=self._tts._speed,
                    lang="en-us",
                )
                return (samples * 32767).astype(np.int16).tobytes()

            pcm = await loop.run_in_executor(None, _synth)
            output_emitter.push(pcm)  # 👈 audio starts playing after first sentence

        output_emitter.flush()