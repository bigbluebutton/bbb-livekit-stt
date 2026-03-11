# BigBlueButton STT Agent for LiveKit

This application provides Speech-to-Text (STT) for BigBlueButton meetings using LiveKit
as their audio bridge.

Supported STT engines:

- **Gladia** — via the official [LiveKit Gladia plugin](https://docs.livekit.io/agents/integrations/stt/gladia/) (default)
- **OpenAI** — via the [LiveKit OpenAI plugin](https://docs.livekit.io/agents/models/stt/openai/); supports the official OpenAI API and any OpenAI-compatible endpoint

## Getting Started

### Environment prerequisites

- Python 3.10+
- A LiveKit instance
- A Gladia API key **or** an OpenAI API key (depending on your chosen STT provider)
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

    **All options ingested by the Gladia and OpenAI STT plugins are exposed via env vars**.

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

Pre-built images are available via GitHub Container Registry as well.

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

> **Note**: OpenAI STT does not support real-time translation. Only the original
> transcript language is returned, matching the user's BBB speech locale.

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
