# CHANGELOG

All notable changes will be documented in this file.
Intermediate pre-release changes will only be registered *separately* in their
respective tag's CHANGELOG.
Final releases will consolidate all intermediate changes in chronological order.

## UNRELEASED

* feat(openai): add OpenAI STT provider support (official and compatible endpoints)
* feat: add GladiaSttAgent provider and factory
* refactor: extract BaseSttAgent and BaseSttConfig into providers/base.py
* refactor: move GladiaConfig to providers package, delete old agent module
* fix: drop transcripts with no resolvable BBB locale instead of publishing a null one
* build(docker): add LOG_LEVEL env var to control the container's log level
* build: declare aiohttp and numpy as direct dependencies
* docs: correct how the OpenAI provider reaches the API and list its caveats
* docs: expand and correct AGENTS.md

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
