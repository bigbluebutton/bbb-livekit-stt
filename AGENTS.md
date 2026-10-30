# AGENTS.md

Single source of truth for agents working on this repository. `CLAUDE.md` is a
symlink to this file — edit this one.

See @README.md for user-facing documentation.

## Project overview

BigBlueButton (BBB) Speech-to-Text agent built on the LiveKit Agents SDK. It joins
LiveKit rooms, subscribes to audio tracks, transcribes speech via configurable STT
providers (Gladia, OpenAI), and publishes transcript events to BBB's backend over
Redis pub/sub.

This is **not** a voice AI agent: there is no LLM, no `Agent` class, no tools, no
handoffs or tasks, and no telephony. LiveKit guidance about workflows, prompt
design, and tool descriptions does not apply here.

## Project structure

All source files live at the **repository root** — there is no `src/` directory.
`main.py` is the entrypoint (see the `Dockerfile` for how it is deployed).

This project uses the `uv` package manager. Always use `uv` to install
dependencies, run the agent, and run tests.

- **`main.py`** — registers the LiveKit worker via `cli.run_app(WorkerOptions)`.
  `entrypoint()` wires up `RedisManager`, the STT agent (via `create_agent()`), and
  the final/interim transcript handlers, and routes Redis messages for locale and
  speech-option changes and user removal. Its `prewarm_fnc` calls
  `prewarm_provider()` so a provider with slow assets loads them once per worker
  process rather than once per room.
  Provider-agnostic: no references to any specific provider.
- **`providers/`** — STT provider abstraction:
  - **`__init__.py`** — factory
    `create_agent(provider, userdata=None) -> BaseSttAgent`, accepting `"gladia"`,
    `"openai"` and `"voxtral-realtime"`, plus `prewarm_provider(provider, userdata)`
    for the worker's prewarm hook. `create_agent()` is the only validator of the
    provider name; `prewarm_provider()` ignores an unknown one so a bad name cannot
    kill the worker before it reports the real error.
  - **`base.py`** — `BaseSttAgent(EventEmitter, ABC)` + `BaseSttConfig`. All
    provider-agnostic logic: room management, track subscription, audio pipeline,
    event emission, `_sanitize_locale()`.
  - **`gladia.py`** — `GladiaSttAgent` + `GladiaConfig` and the `gladia_config`
    singleton. Wraps the official LiveKit Gladia plugin. Confidence filtering via
    `_should_emit()` and translation support via `translation_lang_map`.
  - **`openai.py`** — `OpenAiSttAgent` + `OpenAiConfig`. **Does not use the LiveKit
    OpenAI plugin**: it is a direct `aiohttp` client against
    `{base_url}/v1/audio/transcriptions`, because the plugin's `stream()` speaks a
    Realtime WebSocket protocol that OpenAI-compatible backends generally lack. So
    it overrides `start_transcription_for_user()` and `_run_transcription_pipeline()`
    wholesale, segments audio locally with an RMS silence detector, and raises
    `NotImplementedError` from `_create_stt_stream()`. No confidence filtering, no
    translation.
  - **`voxtral_realtime.py`** — `VoxtralRealtimeSttAgent` + `VoxtralRealtimeConfig`
    and the `voxtral_realtime_config` singleton, for
    [Voxtral Mini Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
    served by a self-hosted [vLLM](https://docs.vllm.ai/). A direct `aiohttp`
    WebSocket client against `{base_url}/realtime?intent=transcription`: vLLM's
    realtime protocol is not the OpenAI Realtime one (flat `session.update`, no
    server-side VAD, `transcription.delta`/`transcription.done` events). Like
    `openai.py` it overrides `start_transcription_for_user()` and
    `_run_transcription_pipeline()` wholesale and raises `NotImplementedError` from
    `_create_stt_stream()`, but it segments speech with a Silero VAD (loaded once
    per worker process via `prewarm()`) and streams audio while a request is open,
    so it emits live interims. No confidence filtering, no translation, and no
    language in either direction — vLLM's realtime API has no field to request
    one and reports none back, so `"auto"` is not usable and a transcript is
    labelled with the locale the participant selected. See the README's caveats.
- **`config.py`** — `RedisConfig` + the `redis_config` singleton, `MetricsConfig` +
  the `metrics_config` singleton, the `stt_provider` env var, env-var helpers
  (`_get_float_env`, `_get_bool_env`, …), and startup config redaction. Provider
  configs live in `providers/`, not here.
- **`metrics.py`** — Prometheus collectors, owned by `SttMetrics`. The `stt_metrics`
  singleton binds to the default registry, which the LiveKit worker's `/metrics`
  endpoint serves. Must not import from `providers/` — providers import it.
- **`redis_manager.py`** — async Redis pub/sub. Publishes `UpdateTranscriptPubMsg`
  to `to-akka-apps-redis-channel`; listens on `from-akka-apps-redis-channel` for
  locale and speech-option change events and `UserLeftMeetingEvtMsg`.
- **`events.py`** — minimal async `EventEmitter`, the base of `BaseSttAgent`.
- **`utils.py`** — `resolve_bbb_locale()` plus coercion helpers for values arriving
  over Redis (`coerce_partial_utterances`, `coerce_min_utterance_length_seconds`).

### Data flow

1. BBB sends `UserSpeechLocaleChangedEvtMsg` over Redis → `main.py` routes it to the
   agent.
2. `start_transcription_for_user()` sanitizes the locale and starts a per-participant
   pipeline (Gladia via `_create_stt_stream()`; OpenAI via its own override).
3. The provider emits transcript events → `_should_emit()` filters them (e.g. Gladia
   confidence thresholds) → the agent emits the survivors.
4. `main.py` adjusts timestamps with the event's `open_time` and resolves the BBB
   locale from its `locale`. `open_time` is when that participant's provider stream
   opened, since providers report offsets within their own stream. `locale` is the
   one the audio was transcribed under; the participant may have changed it since.
   Speech options are read live instead.
5. `RedisManager` publishes `UpdateTranscriptPubMsg` back to BBB.

### Locale handling

BBB uses `<ISO 639-1>-<ISO 3166-1>` (e.g. `en-US`); providers take `<ISO 639-1>`
(e.g. `en`). `BaseSttAgent._sanitize_locale()` converts between them and returns
`str | None` — **`"auto"` sanitizes to `None`**, meaning the participant asked for
server-side detection and providers must omit the language entirely.

`utils.resolve_bbb_locale()` maps a transcript back to a BBB locale, returning `None`
when the language is unknown (auto-detection plus a provider that reports none);
callers must then discard the transcript rather than publish a null locale.
`translation_lang_map` lives in `providers/gladia.py` and is empty for providers
without translation support.

Transcription is enabled only when `UserSpeechLocaleChangedEvtMsg` carries **both** a
`locale` and a `provider`. The html5 client sends both incomplete shapes: the
transcription toggle (`audio-captions/panel/speech/component.tsx`) empties the
locale, and the Audio Settings language select
(`audio-captions/captions/component.tsx`) blanks both. akka-bbb-apps forwards either
verbatim.

`handle_speech_locale_change()` routes any incomplete pair to
`disable_transcription_for_user()`, which stops the session *and* drops the stored
locale/provider. `_on_track_subscribed()` restarts from `participant_settings`, so a
locale left behind would resurrect transcription the user turned off. Whether a
session is running is read from `processing_info`, never from the stored locale.

### Shutdown

LiveKit awaits a job's shutdown callbacks, not its entrypoint: the job process exits
as soon as they return, wherever `start()` got to. So `main.py` registers one that
calls `BaseSttAgent.shutdown()`, which returns once `start()` has torn the agent
down, and then closes Redis. Once shutdown begins, no session starts.

`_cleanup()` stops every session, waits for every pipeline still running and every
stream close the stop path started, then gives transcript handlers up to 5 s. A
fixed pause in its place races the teardown and strands `bbb_stt_active_agents`.

### Adding a new provider

1. Create `providers/<name>.py` with a config dataclass extending `BaseSttConfig`
   and an agent class extending `BaseSttAgent`.
2. Implement `_create_stt_stream()`; optionally override `_should_emit()` and
   `translation_lang_map`. The returned stream must accept `end_input()`, called
   when the audio runs out, and `aclose()`. The pipeline closes it on exit, or
   `stop_transcription_for_user()` does if the pipeline never ran.

   A locale change is **not** an in-place update: `update_locale_for_user()`
   restarts the session. Providers restart their clock when the language changes;
   Gladia opens a fresh live session behind `update_options()`. Transcripts stamped
   against the old stream would land in the past and reuse transcript ids BBB
   already stored.
3. **Handle a `None` locale** in `_create_stt_stream()` — it is how `"auto"`
   reaches a provider. Omitting the language is what enables server-side detection.
4. Register it in the factory in `providers/__init__.py`; if it loads slow assets
   (a local model), give it a `prewarm(userdata)` and dispatch to it from
   `prewarm_provider()` too.
5. `providers/gladia.py` is the reference implementation for stream-based providers;
   `providers/openai.py` for providers that need their own pipeline, and
   `providers/voxtral_realtime.py` for one that also does its own VAD and
   segmentation.

## Setup

Requires Python 3.11+ (`.python-version` pins 3.11: the Silero VAD used by the
`voxtral-realtime` provider pulls in onnxruntime, which dropped its Python 3.10
wheels in 1.24). Copy `.env.example`
to `.env` and fill in at least `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
`LIVEKIT_API_SECRET`, and a provider key — `GLADIA_API_KEY` by default,
`OPENAI_API_KEY` alongside `STT_PROVIDER=openai`, or `VOXTRAL_BASE_URL` alongside
`STT_PROVIDER=voxtral-realtime` (`VOXTRAL_API_KEY` only if the server enforces
one).

## Commands

```bash
uv sync                                              # install dependencies

uv run python3 main.py dev                           # run (development)
uv run python3 main.py start                         # run (production)

uv run pytest tests/ --ignore=tests/integration      # unit tests
uv run pytest tests/test_gladia_agent.py             # single file
uv run pytest tests/test_gladia_agent.py::TestSanitizeLocale::test_lowercases_language_code
uv run pytest tests/ --ignore=tests/integration --cov --cov-report=term-missing

uv run ruff check .                                  # lint
uv run ruff check --fix .
uv run ruff format .
uv run ruff format --check .                         # what CI runs; fails on any delta

docker build . -t bbb-livekit-stt
docker run --network host --rm -it --env-file .env bbb-livekit-stt
```

Maintain code formatting with ruff. Linting changes to unrelated code belong in
separate commits.

## Testing

- pytest with `asyncio_mode = "auto"` (pytest-asyncio).
- Coverage sources: `config`, `events`, `main`, `metrics`, `providers`,
  `redis_manager`, `utils` (see `[tool.coverage.run]` in `pyproject.toml`). CI
  enforces `--cov-fail-under=65`.
- Integration tests are marked `@pytest.mark.integration`, hit the real Gladia and
  OpenAI APIs, and need real keys:
  `GLADIA_API_KEY=<key> uv run pytest tests/integration -m integration`.

Add tests for agent behaviour whenever practical; refer to existing tests in
`tests/`. When changing core behaviour — the provider contract, locale handling,
transcript resolution, or Redis message handling — never guess what will work. Use
test-driven development: write tests for the desired behaviour first, then iterate
until they pass. Do not add tests for input shapes the callers cannot produce.

### Gotchas

- **Bare `uv run pytest` runs the integration tests too.** `testpaths = ["tests"]`
  and there is no default marker filter, so it collects `tests/integration` and
  makes live, billable API calls. Always pass `--ignore=tests/integration` for unit
  runs.
- **Bare imports**: `pythonpath = ["."]` in `pyproject.toml` enables root-level
  imports like `from config import ...`. Moving files into a package means updating
  this and every import.
- **Config singletons**: `redis_config` and `metrics_config` are built at import
  time in `config.py`; `gladia_config`, `openai_config` and
  `voxtral_realtime_config` in their provider modules. All read env vars at import.
  In tests, set env vars before importing or construct the config directly (e.g.
  `GladiaConfig(api_key="fake-key")`).
  `VoxtralRealtimeConfig` additionally needs a `base_url` — the agent's constructor
  raises `ValueError` without one.
- **Metrics run across processes.** LiveKit runs each job in its own process, so
  every `Gauge` in `metrics.py` declares `multiprocess_mode="livesum"`, and
  `PROMETHEUS_MULTIPROC_DIR` must be set as a real env var — `prometheus_client`
  picks its value class at import time, long before `WorkerOptions` could set it.
  Passing `prometheus_multiproc_dir` to `WorkerOptions` instead silently drops the
  worker's own metrics. A killed job's gauges stay until the worker's
  `sweep_dead_processes()` drops them.
- **Metrics tests need a fresh registry.** Construct `SttMetrics(CollectorRegistry())`
  per test and read values with `registry.get_sample_value(name, labels)`. Reusing
  the default registry makes tests order-dependent.
- **`processing_info` entries require `metrics_locale` and `track_sid`.**
  `stop_transcription_for_user()` and `_release_session_slot()` decrement the
  session gauge under `metrics_locale`; `_on_track_unsubscribed()` ends the session
  only for the track in `track_sid`. The pipeline adds `started` once running, so
  `stop_transcription_for_user()` knows whether it must close the provider stream
  itself. Hand-built test entries must include whichever key the path under test
  reads.
- **A pipeline only closes out a `processing_info` entry it still owns.** It
  compares `info["task"]` with `asyncio.current_task()` first. A session restarted
  in the same event-loop turn (disable→enable, or a locale change) already holds
  the slot when the cancelled task unwinds. That check also balances the session
  gauge; see `_release_session_slot()`. Tests that hand-build an entry and `await`
  the pipeline directly must set `"task": asyncio.current_task()`, or the cleanup
  will not fire.
- **Test patch targets**: patch at the import location — `"providers.gladia.GladiaSTT"`,
  not `"livekit.plugins.gladia.STT"`; `"providers.base.rtc.AudioStream"` for the
  shared audio pipeline, `"providers.openai.rtc.AudioStream"` and
  `"providers.voxtral_realtime.rtc.AudioStream"` for the providers that run their
  own.

## LiveKit documentation

LiveKit Agents evolves quickly; always consult the latest documentation. LiveKit
offers an MCP server for browsing and searching it — recommend installing it if the
developer has not (details at https://docs.livekit.io/mcp):

- Claude Code: `claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp`
- Codex: `codex mcp add --url https://docs.livekit.io/mcp livekit-docs`
- Gemini: `gemini mcp add --transport http livekit-docs https://docs.livekit.io/mcp`
- Cursor: [install link](https://cursor.com/en-US/install-mcp?name=livekit-docs&config=eyJ1cmwiOiJodHRwczovL2RvY3MubGl2ZWtpdC5pby9tY3AifQ%3D%3D)

The LiveKit CLI (`lk`) can be used for room management and similar tasks, with user
approval. See https://docs.livekit.io/home/cli.

## CHANGELOG

Always update `CHANGELOG.md` when committing any of the following types of change:

- feat
- fix
- build
- breaking changes
- docs

Changelog entries MUST be the commit message title and should be added in
chronological order under the UNRELEASED section. Changes should be aggregated by
commit type; see above list of prefixes and follow that order.

## Commit message guidelines

- Commit message titles should be structured with convetional-commits: https://www.conventionalcommits.org/en/v1.0.0/
- Meaningful message
```
The body should provide a meaningful commit message, which:
. explains the problem the change tries to solve, i.e. what is wrong
  with the current code without the change.
. justifies the way the change solves the problem, i.e. why the
  result with the change is better.
. alternate solutions considered but discarded, if any.
```
- Make separate commits for logically separate changes
```
Unless your patch is really trivial, you should not be sending
out a patch that was generated between your working tree and
your commit head.  Instead, always make a commit with complete
commit message and generate a series of patches from your
repository.  It is a good discipline.

Give an explanation for the change(s) that is detailed enough so
that people can judge if it is good thing to do, without reading
the actual patch text to determine how well the code does what
the explanation promises to do.

If your description starts to get too long, that's a sign that you
probably need to split up your commit to finer grained pieces.
That being said, patches which plainly describe the things that
help reviewers check the patch, and future maintainers understand
the code, are the most beautiful patches.  Descriptions that summarize
the point in the subject well, and describe the motivation for the
change, the approach taken by the change, and if relevant how this
differs substantially from the prior version, are all good things
to have.
```
- Present tense
```
The problem statement that describes the status quo is written in the
present tense.  Write "The code does X when it is given input Y",
instead of "The code used to do Y when given input X".  You do not
have to say "Currently"---the status quo in the problem statement is
about the code _without_ your change, by project convention.
```
- Imperative mood
```
Describe your changes in imperative mood, e.g. "make xyzzy do frotz"
instead of "[This patch] makes xyzzy do frotz" or "[I] changed xyzzy
to do frotz", as if you are giving orders to the codebase to change
its behavior.  Try to make sure your explanation can be understood
without external resources. Instead of giving a URL to a mailing list
archive, summarize the relevant points of the discussion.
```
