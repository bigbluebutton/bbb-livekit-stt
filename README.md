# BigBlueButton STT Agent for LiveKit

This application provides Speech-to-Text (STT) for BigBlueButton meetings using LiveKit
as their audio bridge.

Supported STT engines:

- **Gladia** — via the official [LiveKit Gladia plugin](https://docs.livekit.io/agents/integrations/stt/gladia/) (default)
- **OpenAI** — via a direct REST client against `/v1/audio/transcriptions`; supports the official OpenAI API and any OpenAI-compatible endpoint
- **Voxtral Realtime** — [Mistral Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) served by a self-hosted [vLLM](https://docs.vllm.ai/) instance via its realtime WebSocket API

## Getting Started

### Environment prerequisites

- Python 3.11+
- A LiveKit instance
- A Gladia API key, an OpenAI API key, **or** a vLLM server hosting Voxtral (depending on your chosen STT provider)
- uv:
  - See installation instructions: https://docs.astral.sh/uv/getting-started/installation/

### Installing

1.  **Clone the repository:**

    ```bash
    git clone git@github.com:bigbluebutton/bbb-livekit-stt.git
    cd bbb-livekit-stt
    ```

2.  **Install the dependencies:**

    ```bash
    uv sync
    ```

4.  **Configure environment variables:**

    Copy the example `.env` file:

    ```bash
    cp .env.example .env
    ```

    Now, edit the `.env` file and fill _at least_ the following environment vars:

    ```
    LIVEKIT_URL=...
    LIVEKIT_API_KEY=...
    LIVEKIT_API_SECRET=...

    # For Gladia (default provider):
    GLADIA_API_KEY=...

    # For OpenAI (set STT_PROVIDER=openai):
    # STT_PROVIDER=openai
    # OPENAI_API_KEY=...
    ```

    Feel free to check `.env.example` for any other configurations of interest.

    **All options ingested by the Gladia STT plugin are exposed via env vars**. The
    OpenAI provider takes `OPENAI_API_KEY`, `OPENAI_STT_MODEL` and `OPENAI_BASE_URL`
    (see below).

### Running

The agent is run using the command-line interface provided by the `livekit-agents`
library. The necessary environment variables will be  picked up automatically.

Once started, the worker will connect to your LiveKit server and wait to be assigned
to rooms. By default, the LiveKit server will dispatch a job to the worker for every
new room created. The agent will then join the room, start listening to audio tracks,
and generate transcription events when required.

#### Development

For development, use the `dev` command.

```bash
uv run python3 main.py dev
```

#### Production

For production, use the `start` command.

```bash
uv run python3 main.py start
```

#### Docker

Build the image:

```bash
docker build . -t bbb-livekit-stt
```

Run:

```bash
docker run --network host --rm -it --env-file .env bbb-livekit-stt
```

The container's log level is controlled by the `LOG_LEVEL` env var (`TRACE`,
`DEBUG`, `INFO`, `WARN`, `ERROR` or `CRITICAL`; defaults to `INFO`). To run the
agent with debug logging:

```bash
docker run --network host --rm -it --env-file .env -e LOG_LEVEL=DEBUG bbb-livekit-stt
```

Pre-built images are available via GitHub Container Registry as well.

### Metrics

The agent exposes Prometheus metrics through the LiveKit worker's built-in
endpoint. Metrics are **opt-in**: set `BBB_STT_PROMETHEUS_PORT` to enable them.

```bash
BBB_STT_PROMETHEUS_PORT=8082
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-multiproc
```

Scrape `http://<host>:8082/metrics`.

`PROMETHEUS_MULTIPROC_DIR` is required, not optional. LiveKit runs each room job
in its own process, and per-session metrics only reach the endpoint through this
shared directory. It must be a real environment variable: `prometheus_client`
decides whether to use multiprocess mode when it is first imported, which happens
before the worker could set it. The Docker image sets it already.

With Docker:

```bash
docker run --network host --rm -it --env-file .env \
  -e BBB_STT_PROMETHEUS_PORT=8082 bbb-livekit-stt
```

The endpoint also serves LiveKit's own `lk_agents_*` metrics (active jobs, worker
load, child process count).

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `bbb_stt_active_sessions` | Gauge | `provider`, `locale` | Active transcription sessions, one per subscribed microphone track |
| `bbb_stt_active_agents` | Gauge | `provider` | Running agents, one per room job |
| `bbb_stt_session_starts_total` | Counter | `provider` | Sessions successfully started |
| `bbb_stt_session_start_failures_total` | Counter | `provider`, `reason` | Sessions that could not be established |
| `bbb_stt_agent_start_failures_total` | Counter | `provider`, `reason` | Agents that could not be established |
| `bbb_stt_transcripts_total` | Counter | `provider`, `event_type` | Transcripts that passed filtering and were sent to BBB |
| `bbb_stt_transcripts_discarded_total` | Counter | `provider`, `event_type`, `reason` | Transcripts dropped before publication |
| `bbb_stt_transcript_confidence` | Histogram | `provider`, `event_type` | Confidence of published transcripts (Gladia only) |
| `bbb_stt_utterance_duration_seconds` | Histogram | `provider`, `event_type` | Duration of transcribed utterances |
| `bbb_stt_provider_audio_seconds_total` | Counter | `provider` | Audio seconds sent to the provider, as counted by the Gladia plugin (Gladia only) |
| `bbb_stt_language_detection_mismatch_total` | Counter | `provider` | Transcripts whose language differs from the requested one |
| `bbb_stt_transcript_publish_failures_total` | Counter | `provider` | Transcripts lost because publishing to Redis failed |

Caveats worth knowing:
- No metric carries a meeting or user label, to keep cardinality bounded. The
  logs carry per-meeting detail.
- **`bbb_stt_session_start_failures_total{reason="no_audio_track"}` is usually
  benign.** BBB routinely sends the speech-locale change before the participant
  publishes a microphone track, and the session starts moments later. Alert on
  its rate relative to `bbb_stt_session_starts_total`, not on the raw count.
- **`bbb_stt_transcripts_total` counts transcripts that were *sent*, not
  confirmed.** A transcript is counted even when the Redis publish fails, so the
  confidence histogram has no gaps during a Redis outage. Subtract
  `bbb_stt_transcript_publish_failures_total` for what BBB received.
- **`bbb_stt_provider_audio_seconds_total` undercounts.** The Gladia plugin
  reports audio every 5 s or so. Audio sent after the last report can be lost
  when a session ends: under 5 s normally, more after a slow Gladia connection.
- **A job killed with `SIGKILL` leaves its gauge value behind.** `prometheus_client`
  drops a dead process's gauge contribution only when `mark_process_dead` is
  called, and `livekit-agents` never calls it. Clean exits, exceptions and
  cancellations decrement correctly; an OOM kill does not. The stale value
  clears when the worker restarts. LiveKit's own `lk_agents_active_job_count` is
  affected identically, so `bbb_stt_active_agents` exceeding it indicates leaked
  series.
- **A locale change restarts the participant's session.** It appears as a stop
  and a start: `bbb_stt_session_starts_total` advances and the session gauge moves
  between locale labels. A change the provider cannot honour therefore shows up as
  a `bbb_stt_session_start_failures_total` tick, not as a session left running in
  the old locale.
- `bbb_stt_transcript_confidence` is only populated for Gladia as of now. Providers
  that do not report transcript confidence (e.g.: OpenAI) do not populate this histogram.

### OpenAI STT provider

Set `STT_PROVIDER=openai` to use OpenAI STT instead of Gladia.

**Official OpenAI API:**

```bash
STT_PROVIDER=openai
OPENAI_API_KEY=your-key
# OPENAI_STT_MODEL=gpt-4o-transcribe  # default; use "whisper-1" for classic Whisper
```

**OpenAI-compatible endpoint** (e.g. a self-hosted Whisper server):

```bash
STT_PROVIDER=openai
OPENAI_API_KEY=any-value
OPENAI_BASE_URL=http://your-server:8000
OPENAI_STT_MODEL=your-model-name
```

Some caveats apply to this provider:

- It is not a streaming provider. Audio is segmented locally by silence detection
  and each segment is posted to `/v1/audio/transcriptions`, so transcripts arrive
  once an utterance ends rather than continuously.
- It does not support real-time translation. Only the original transcript
  language is returned, matching the user's BBB speech locale.
- The `auto` speech locale is not usable with it. The agent requests
  `response_format=json`, whose payload carries only the transcribed text, so
  there is no detected language to map back to a BBB locale and the transcript is
  discarded with a warning. Set an explicit locale in BBB when using OpenAI STT.

### Voxtral Realtime STT provider

Set `STT_PROVIDER=voxtral-realtime` to use a self-hosted
[Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
model served by [vLLM](https://docs.vllm.ai/). The agent streams each
participant's audio over vLLM's realtime WebSocket API, gated by a local
Silero VAD, and emits live interim captions from the model's incremental
deltas.

```bash
STT_PROVIDER=voxtral-realtime
VOXTRAL_BASE_URL=https://your-vllm-server:8000/v1   # required
VOXTRAL_API_KEY=your-key                            # if your server enforces one
# VOXTRAL_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602  # default
```

VAD and segmentation tuning options (silence duration, pre-roll, split
overlap, max segment length) are documented in `.env.example`.

Some caveats apply to this provider:

- It does not support real-time translation. Only the original transcript
  language is returned, matching the user's BBB speech locale.
- The `auto` speech locale is not usable with it, and the participant's chosen
  locale never reaches the model either. vLLM's realtime API carries no language
  in either direction: `session.update` accepts only `model`, and
  `transcription.delta` / `transcription.done` report only text. The limitation
  is in the model's streaming prompt format, not just the API — a target
  language is encoded as a `lang:<code>` prefix only in the offline
  transcription format, never in the streaming one. Voxtral Realtime therefore
  always detects the language itself, transcripts are labelled with the locale
  the participant selected, and under `auto` there is no language at all, so
  every transcript — interim and final — is discarded with a warning. Set an
  explicit locale in BBB when using Voxtral Realtime STT.

### Development

#### Testing

Run the unit tests:

```bash
uv run pytest tests/ --ignore=tests/integration
```

Run with coverage:

```bash
uv run pytest tests/ --ignore=tests/integration --cov --cov-report=term-missing
```

Integration tests require a real API key and make live requests to the STT service.

For Gladia, set `GLADIA_API_KEY` and run:

```bash
GLADIA_API_KEY=your-key uv run pytest tests/integration -m integration
```

For OpenAI, set `OPENAI_API_KEY` and run:

```bash
OPENAI_API_KEY=your-key uv run pytest tests/integration -m integration
```

#### Linting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting. To check for issues:

```bash
uv run ruff check .
```

To automatically fix fixable issues:

```bash
uv run ruff check --fix .
```

To format the code:

```bash
uv run ruff format .
```
