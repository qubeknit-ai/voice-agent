import os
import logging
import time
import tempfile
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents import llm as lk_llm

from livekit.plugins import google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# ── local EdgeTTS to replace slow Gemini TTS ──────────────────────────────
from plugins.edge_tts_plugin import EdgeTTS

load_dotenv(".env")
# CREDENTIALS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


logger = logging.getLogger(__name__)

def get_credentials_file() -> str | None:
    """
    Railway: set GOOGLE_CREDENTIALS_JSON env var with the full JSON content.
    Local:   set GOOGLE_APPLICATION_CREDENTIALS env var with file path.
    """
    # Railway / production — JSON string in env var
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if creds_json:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        tmp.write(creds_json)
        tmp.flush()
        return tmp.name

    # Local — file path in env var
    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


CREDENTIALS_FILE = get_credentials_file()



class Assistant(Agent):
    def __init__(self) -> None:
        # ✅ pre-seed chat context — Gemini rejects empty contents
        initial_ctx = lk_llm.ChatContext()
        initial_ctx.add_message(role="user", content="[session started]")

        super().__init__(
            instructions="""You are Alie, hotel assistant at White Meadow Hotel, Hunza Valley, Pakistan.
            Rules: Max 2 sentences per reply. No formatting. Spoken words only.
            Hotel: mountain views, near Baltit Fort, Attabad Lake, Rakaposhi peaks.""",
            chat_ctx=initial_ctx,
        )

    async def on_enter(self):  # ✅ correct lifecycle hook
        await self.session.generate_reply(
            instructions="Greet the user warmly in one short sentence. Max 12 words."
        )


server = AgentServer()


@server.rtc_session(agent_name="call_agent")
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=google.STT(
            model="chirp_2",
            location="us-central1",
            credentials_file=CREDENTIALS_FILE,
            languages=["en-US"],
            # ✅ voice activity timeout — cuts Chirp latency from 2.4s → ~0.5s
            spoken_punctuation=False,
        ),
        llm=google.LLM(
            model="gemini-2.5-flash",
            max_output_tokens=80,          # ✅ stop over-generating
        ),
        tts=google.TTS(                        # ✅ swapped from GeminiTTS
            voice_name="en-US-Chirp3-HD-Aoede",
            credentials_file=CREDENTIALS_FILE,
        ),
        vad=silero.VAD.load(
            min_silence_duration=0.2,      # ✅ faster end-of-speech detection
            activation_threshold=0.5,
        ),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        min_endpointing_delay=0.05,        # ✅ was missing — cuts end_of_turn
        max_endpointing_delay=0.3,
        min_interruption_duration=0.3,
        min_interruption_words=1,
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None
    _eou_time: float | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics, _eou_time
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
            _eou_time = time.time()        # ✅ correct timestamp capture
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage Summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking" and _eou_time:
            elapsed = time.time() - _eou_time   # ✅ correct elapsed calc
            logger.info(f"Time to first audio: {elapsed:.3f}s")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)