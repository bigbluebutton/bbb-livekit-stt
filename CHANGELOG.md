# CHANGELOG

All notable changes will be documented in this file.
Intermediate pre-release changes will only be registered *separately* in their
respective tag's CHANGELOG.
Final releases will consolidate all intermediate changes in chronological order.

## UNRELEASED

* feat(tests): add unit and integration tests with pytest
* feat(tests): add coverage reporting with pytest-cov
* feat(tests): add tests for v0.2.0 changes (utils coercions, config redaction, on_track_subscribed fix, new defaults)
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
