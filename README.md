# BigBlueButton STT Agent for LiveKit

This application provides Speech-to-Text (STT) for BigBlueButton meetings using LiveKit
as their audio bridge.

Initially, the only supported STT engine is Gladia through the official  [LiveKit Gladia Plugin](https://docs.livekit.io/agents/integrations/stt/gladia/).

It'll be expanded in the future to support other STT plugins from the LiveKit Agents
ecosystem.

## Getting Started

### Environment prerequisites

- Python 3.10+
- A LiveKit instance
- A Gladia API key
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

    # Gladia API Key
    GLADIA_API_KEY=...
    ```

    Feel free to check `.env.example` for any other configurations of interest.

    **All options ingested by the Gladia STT plugin are exposed via env vars**.

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

#### Docker (locally)

Build the image:

```bash
docker build . -t bbb-livekit-stt
```

Run:

```bash
docker run --network host --rm -it --env-file .env bbb-livekit-stt
