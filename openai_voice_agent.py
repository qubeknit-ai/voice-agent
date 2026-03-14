from dotenv import load_dotenv
import logging
import time

from livekit import agents, rtc
from livekit.agents import AgentServer,AgentSession, Agent, room_io
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics

from livekit.plugins import noise_cancellation, silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")
logger = logging.getLogger(__name__)


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are Alie, hotel assistant at White Meadow Hotel, Hunza Valley, Pakistan.
                Rules: Max 2 sentences per reply. No formatting. Spoken words only.
                Hotel: mountain views, near Baltit Fort, Attabad Lake, Rakaposhi peaks.""",
        )

server = AgentServer()

@server.rtc_session(agent_name="call_agent")
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=openai.STT(
            model="whisper-1"
        ),
        llm=openai.LLM(
            model="gpt-4o-mini",
        ),
        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="ash",
            instructions="Speak in a warm, welcoming, and friendly tone suited for hotel customer service.",  # 👈 optional style control
        ),
        vad=silero.VAD.load(
            min_silence_duration=0.2,      # 👈 declare end after 200ms silence
            activation_threshold=0.5,
        ),
        turn_detection=MultilingualModel(),

        # ✅ Fix 1 — reduce end_of_turn delay (was default ~1.5s)
        min_endpointing_delay=0.1,   # detect silence faster
        max_endpointing_delay=0.4,   # don't wait more than 0.8s

        # ✅ Fix 2 — start generating reply before turn is fully confirmed
        preemptive_generation=True,

        # ✅ Fix 3 — interruption tuning
        min_interruption_duration=0.3,
        min_interruption_words=2,
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")  # ✅ fixed typo: metrices → metrics
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage Summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                elapsed = time.time() - last_eou_metrics.end_of_utterance_delay
                logger.info(f"Time to first audio: {elapsed:.3f}s")

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony() if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP else noise_cancellation.BVC(),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)