# Remix Pipeline

An audio-processing pipeline with persistent job state, quality checks, retries, and human review. Analyze tracks, separate stems, process vocals, generate instrumentation, and align a rough arrangement. Jobs can pause for approval and resume from disk.

## Getting started

Run the state and checkpoint example with Python 3.10+:

```sh
python examples/checkpoint_demo.py
```

It advances a local job through a review checkpoint and reloads the approved state. No audio files or account setup needed.

## Components

| Layer | Components |
| --- | --- |
| Execution | JSON-backed state machine, stage registry, retry accounting, validation, review checkpoints |
| Audio | Tempo/key/structure analysis, Demucs stem separation, vocal processing, instrumental generation, time/pitch alignment |
| Tests | Generated audio fixtures and mocked network/subprocess calls |

Final mixing, mastering, and packaging are unfinished. Individual stages are callable; full orchestration still needs stage-mapping and validator integration. `features.json` tracks the remaining work.

## Development

Python 3.10+ and FFmpeg are needed for the audio components:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Stem separation also requires `python -m pip install -e '.[separation]'`. For instrument generation, export `APIPASS_KEY` in your environment. See `.env.example`; `.env` files are not loaded automatically.

## Code map

- `harness/`: state, validation, retries, review, and dispatch
- `agents/`: audio and data stages
- `tests/`: fixtures and integration checks
- `examples/`: persistent-state example

## License

[MIT](LICENSE)
