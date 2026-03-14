from __future__ import annotations
import asyncio
import io
from livekit.agents import tts, utils
import logging

logger = logging.getLogger(__name__)

# ── Available voices ───────────────────────────────────────────────────────
# en-US-AriaNeural     → warm, conversational (recommended for hotel)
# en-US-JennyNeural    → friendly, clear
# en-US-GuyNeural      → male, neutral
# en-GB-SoniaNeural    → British female, professional
# en-GB-RyanNeural     → British male, polished
# en-AU-NatashaNeural  → Australian female, cheerful
# ──────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class EdgeTTS(tts.TTS):
    def __init__(
        self,
        voice: str = "en-US-AriaNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._volume = volume

    def synthesize(self, text: str, *, conn_options=None) -> tts.ChunkedStream:
        return EdgeStream(tts_instance=self, text=text, conn_options=conn_options)


class EdgeStream(tts.ChunkedStream):
    def __init__(self, *, tts_instance: EdgeTTS, text: str, conn_options=None):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(
            self._input_text,
            self._tts._voice,
            rate=self._tts._rate,
            pitch=self._tts._pitch,
            volume=self._tts._volume,
        )

        # collect all mp3 chunks first
        mp3_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buf.write(chunk["data"])

        mp3_bytes = mp3_buf.getvalue()
        if not mp3_bytes:
            logger.error("edge-tts returned no audio")
            return

        # decode mp3 → pcm in executor
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, _mp3_to_pcm, mp3_bytes)

        if not pcm:
            logger.error("mp3 decode failed")
            return

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()


def _mp3_to_pcm(mp3_bytes: bytes) -> bytes:
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return seg.raw_data