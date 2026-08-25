# CHANGELOG

All notable changes will be documented in this file.
Intermediate pre-release changes will only be registered *separately* in their
respective tag's CHANGELOG.
Final releases will consolidate all intermediate changes in chronological order.

## UNRELEASED

* feat(openai): add OpenAI STT provider support (official and compatible endpoints)
* feat: add GladiaSttAgent provider and factory
* feat: add Prometheus collectors for the STT agent
* feat: expose Prometheus metrics behind BBB_STT_PROMETHEUS_PORT
* feat: instrument session lifecycle across the STT providers
* feat: instrument transcript publication and discards
* refactor: extract BaseSttAgent and BaseSttConfig into providers/base.py
* refactor: move GladiaConfig to providers package, delete old agent module
* fix: drop transcripts with no resolvable BBB locale instead of publishing a null one
* fix: report Redis connect and publish failures to callers
* fix: restart transcription when a speech locale is reassigned
* fix: keep a restarted transcription session's processing_info entry
* fix: return the session gauge when a transcription pipeline ends on its own
* fix: transcribe a participant's microphone track only
* fix: close the provider streams when a transcription session ends
* build(docker): add LOG_LEVEL env var to control the container's log level
* build: declare aiohttp and numpy as direct dependencies
* docs: correct how the OpenAI provider reaches the API and list its caveats
* docs: expand and correct AGENTS.md
* docs: document Prometheus metrics and their operational caveats

## v0.3.0

* feat(tests): add unit and integration tests with pytest
* feat(tests): add coverage reporting with pytest-cov
* feat(tests): add tests for v0.2.0 changes (utils coercions, config redaction, on_track_subscribed fix, new defaults)
* fix: handle "auto" locale to prevent invalid language code sent to Gladia
* build: add GitHub Actions workflow for running tests

## v0.2.0

* feat(stt): support INTERIM transcriptions
* feat: add filtering based on Gladia confidence score
* feat: add env var mappings for remaining Gladia options
* fix: interpret minUtteranceLength as seconds for interim transcripts
* fix: normalize transcript timestamps
* refactor: adjust fallback/default Gladia values
* build: livekit-agents[gladia]~=1.4
* build: add docker image build and publish workflow
* build: add app linting workflow

## v0.1.0

* Initial release
