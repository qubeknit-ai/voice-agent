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

load_dotenv(".env")

logger = logging.getLogger(__name__)


def get_credentials_file() -> str | None:
    """
    Railway / Render:
    Set GOOGLE_CREDENTIALS_JSON env var containing full JSON.

    Local:
    Set GOOGLE_APPLICATION_CREDENTIALS with file path.
    """

    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")

    if creds_json:
        path = "/tmp/google_creds.json"
        with open(path, "w") as f:
            f.write(creds_json)
        return path

    return os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


CREDENTIALS_FILE = get_credentials_file()


class Assistant(Agent):
    def __init__(self) -> None:

        # Gemini requires non-empty chat context
        initial_ctx = lk_llm.ChatContext()
        initial_ctx.add_message(role="user", content="[session started]")

        super().__init__(
            instructions="""
You are Alie, hotel assistant at White Meadow Hotel, Hunza Valley, Pakistan.
Rules: Max 2 sentences per reply. No formatting. Spoken words only.
Hotel: mountain views, near Baltit Fort, Attabad Lake, Rakaposhi peaks.
""",
            chat_ctx=initial_ctx,
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions="Greet the user warmly in one short sentence. Max 12 words."
        )


server = AgentServer()


@server.rtc_session(agent_name="call_agent")
async def entrypoint(ctx: agents.JobContext):

    session = AgentSession(

        # ── Speech-to-Text ─────────────────────────
        stt=google.STT(
            model="chirp_2",
            location="us-central1",
            credentials_file=CREDENTIALS_FILE,
            languages=["en-US"],
            spoken_punctuation=False,
        ),

        # ── Gemini LLM ─────────────────────────────
        llm=google.LLM(
            model="gemini-2.5-flash",
            max_output_tokens=80,
        ),

        # ── Text-to-Speech ─────────────────────────
        tts=google.TTS(
            voice_name="en-US-Chirp3-HD-Aoede",
            credentials_file=CREDENTIALS_FILE,
        ),

        # ── Voice Activity Detection ───────────────
        vad=silero.VAD.load(
            min_silence_duration=0.2,
            activation_threshold=0.5,
        ),

        # Turn detection removed (huge RAM saver)

        preemptive_generation=True,
        min_endpointing_delay=0.05,
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
            _eou_time = time.time()

        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage Summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking" and _eou_time:
            elapsed = time.time() - _eou_time
            logger.info(f"Time to first audio: {elapsed:.3f}s")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )


if __name__ == "__main__":
    import threading
    from http.server import HTTPServer, BaseHTTPRequestHandler

    # Start LiveKit agent in background
    def run_agent():
        agents.cli.run_app(server)

    threading.Thread(target=run_agent, daemon=True).start()

    # Minimal HTTP server so Render detects a port
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"LiveKit agent running")

    port = int(os.environ.get("PORT", 10000))
    httpd = HTTPServer(("0.0.0.0", port), Handler)

    print(f"Web server running on port {port}")
    httpd.serve_forever()