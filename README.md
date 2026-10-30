# Remix Pipeline

An experimental audio-processing pipeline built around persistent job state, quality gates, retries, and human review. It combines conventional audio processing with an API-backed music-generation stage.

The interesting part is the execution infrastructure: a track can advance through multiple stages, pause for approval, and resume from stored state instead of restarting from scratch.

## Try the offline state demo

Python 3.10+; no account, audio files, or third-party packages required:

```sh
python examples/checkpoint_demo.py
```

The demo creates temporary synthetic job state, pauses at a review checkpoint, approves it, and reloads the result from disk. It does not download music or call a generation API.

## Implemented pieces

| Layer | Components |
| --- | --- |
| Execution | JSON-backed state machine, stage registry, error classification/retry accounting, validation dispatch, human checkpoint storage |
| Audio | Tempo/key/structure analysis, Demucs stem separation, vocal processing, instrumental generation adapter, time/pitch alignment and rough arrangement |
| Verification | Synthetic-audio tests and mocked network/subprocess tests |

`features.json` records the implementation boundary: 12 of 15 planned components are marked implemented. Final mixing, mastering, and packaging are unfinished. Stage functions are individually callable; a complete, validated end-to-end command is not provided. Orchestrator stage mapping and audio-validator registration need integration work before a full run.

## Development

Python 3.10+ and FFmpeg are needed for the audio components. Use a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Stem separation additionally requires `python -m pip install -e '.[separation]'`. Instrument generation requires an `APIPASS_KEY` supplied through the environment and can incur charges; it is not needed for the demo or mocked tests. `.env.example` documents the variable; this package does not automatically load `.env` files.

## Code map

- `harness/`: state, validation, retry accounting, review checkpoints, and dispatch
- `agents/`: individual audio/data stage implementations
- `tests/`: offline fixtures, mocked integrations, and synthetic audio
- `examples/`: a dependency-free persistent-state demo

Only use audio you own or are authorized to process. No commercial tracks, stems, trained models, or API credentials are included. Third-party tools and services have their own terms and licenses.

## License

MIT; see [LICENSE](LICENSE)
