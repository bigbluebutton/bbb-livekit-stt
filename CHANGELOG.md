# CHANGELOG

All notable changes will be documented in this file.
Intermediate pre-release changes will only be registered *separately* in their
respective tag's CHANGELOG.
Final releases will consolidate all intermediate changes in chronological order.

## UNRELEASED

* feat(openai): add OpenAI STT provider support (official and compatible endpoints)
* feat: add GladiaSttAgent provider and factory
* refactor: move GladiaConfig to providers package, delete old agent module
* feat(tests): add unit and integration tests with pytest
* feat(tests): add coverage reporting with pytest-cov
* feat(tests): add tests for v0.2.0 changes (utils coercions, config redaction, on_track_subscribed fix, new defaults)
* feat(voxtral): Voxtral Realtime STT provider with concurrent streaming
* feat(voxtral): replace RMS VAD with Silero neural VAD on Python 3.11
* fix(voxtral): reduce word loss at max-buffer segment boundaries
* fix(voxtral): recover from reader failures and flush segments on teardown
* fix(voxtral): drop redundant bare commit at segment close, detect done/segment desync
* fix(voxtral): replay a longer overlap when reopening after a max-buffer split
* fix(voxtral): keep replacement pipeline tracked across locale-change restarts
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
