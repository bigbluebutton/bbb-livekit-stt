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
  speech-option changes. Provider-agnostic: no references to any specific provider.
- **`providers/`** — STT provider abstraction:
  - **`__init__.py`** — factory `create_agent(provider) -> BaseSttAgent`, accepting
    `"gladia"` and `"openai"`.
  - **`base.py`** — `BaseSttAgent(EventEmitter, ABC)` + `BaseSttConfig`. All
    provider-agnostic logic: room management, track subscription, audio pipeline,
    event emission, `_sanitize_locale()`.
  - **`gladia.py`** — `GladiaSttAgent` + `GladiaConfig` and the `gladia_config`
    singleton. Wraps the official LiveKit Gladia plugin. Confidence filtering via
    `_should_emit()`, in-place locale updates via `stream.update_options()`, and
    translation support via `translation_lang_map`.
  - **`openai.py`** — `OpenAiSttAgent` + `OpenAiConfig`. **Does not use the LiveKit
    OpenAI plugin**: it is a direct `aiohttp` client against
    `{base_url}/v1/audio/transcriptions`, because the plugin's `stream()` speaks a
    Realtime WebSocket protocol that OpenAI-compatible backends generally lack. So
    it overrides `start_transcription_for_user()` and `_run_transcription_pipeline()`
    wholesale, segments audio locally with an RMS silence detector, raises
    `NotImplementedError` from `_create_stt_stream()`, and restarts the pipeline in
    `_update_stream_locale()`. No confidence filtering, no translation.
- **`config.py`** — `RedisConfig` + the `redis_config` singleton, the `stt_provider`
  env var, env-var helpers (`_get_float_env`, `_get_bool_env`, …), and startup
  config redaction. Provider configs live in `providers/`, not here.
- **`redis_manager.py`** — async Redis pub/sub. Publishes `UpdateTranscriptPubMsg`
  to `to-akka-apps-redis-channel`; listens on `from-akka-apps-redis-channel` for
  locale and speech-option change events.
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
4. `main.py` adjusts timestamps with `open_time` and calls `resolve_bbb_locale()` to
   map the transcript back to a BBB locale.
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

### Adding a new provider

1. Create `providers/<name>.py` with a config dataclass extending `BaseSttConfig`
   and an agent class extending `BaseSttAgent`.
2. Implement `_create_stt_stream()` and `_update_stream_locale()`; optionally
   override `_should_emit()` and `translation_lang_map`.
3. **Handle a `None` locale** in both — it is how `"auto"` reaches a provider.
   Omitting the language is what enables server-side detection.
4. Register it in the factory in `providers/__init__.py`.
5. `providers/gladia.py` is the reference implementation for stream-based providers;
   `providers/openai.py` for providers that need their own pipeline.

## Setup

Requires Python 3.10+ (`.python-version` pins 3.10). Copy `.env.example` to `.env`
and fill in at least `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and a
provider key — `GLADIA_API_KEY` by default, or `OPENAI_API_KEY` alongside
`STT_PROVIDER=openai`.

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
- Coverage sources: `config`, `events`, `providers`, `redis_manager`, `utils`. CI
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
- **Config singletons**: `redis_config` is built at import time in `config.py`;
  `gladia_config` and `openai_config` in their provider modules. All read env vars
  at import. In tests, set env vars before importing or construct the config
  directly (e.g. `GladiaConfig(api_key="fake-key")`).
- **Test patch targets**: patch at the import location — `"providers.gladia.GladiaSTT"`,
  not `"livekit.plugins.gladia.STT"`; `"providers.base.rtc.AudioStream"` for the
  shared audio pipeline, `"providers.openai.rtc.AudioStream"` for OpenAI's own.

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
