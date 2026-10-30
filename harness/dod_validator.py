from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


class DODValidator:
    def __init__(self, project_root: Path):
        self._root = Path(project_root)

    def validate(self, track_id: str, stage: str, **kwargs) -> ValidationResult:
        validator = self._validators.get(stage)
        if validator is None:
            return ValidationResult(passed=False, reasons=[f"Unknown stage: {stage}"])
        return validator(self, track_id, **kwargs)

    def _validate_scrape(self, track_id: str, **kwargs) -> ValidationResult:
        reasons = []
        manifest_path = self._root / "data" / "tracks_manifest.json"
        if not manifest_path.exists():
            return ValidationResult(passed=False, reasons=["Manifest file missing"])

        data = json.loads(manifest_path.read_text())
        tracks = data.get("tracks", [])

        if len(tracks) < 10:
            reasons.append(f"Too few entries: {len(tracks)} (need >= 10)")

        required = {"rank", "title", "artist", "track_id", "status"}
        for i, t in enumerate(tracks):
            missing = required - set(t.keys())
            if missing:
                reasons.append(f"Entry {i}: missing fields {missing}")
                break

        ids = [t.get("track_id") for t in tracks]
        if len(ids) != len(set(ids)):
            reasons.append("Duplicate track_ids found")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_download(self, track_id: str, duration_seconds: float = 0, **kwargs) -> ValidationResult:
        reasons = []
        mp3_path = self._root / "raw" / f"{track_id}.mp3"

        if not mp3_path.exists():
            return ValidationResult(passed=False, reasons=[f"MP3 missing: {mp3_path}"])

        size = mp3_path.stat().st_size
        if size < 3 * 1024 * 1024:
            reasons.append(f"File too small: {size} bytes (need > 3MB)")

        if duration_seconds <= 90:
            reasons.append(f"Duration too short: {duration_seconds}s (need > 90s)")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_analyze(self, track_id: str, **kwargs) -> ValidationResult:
        reasons = []
        path = self._root / "analysis" / f"{track_id}_analysis.json"

        if not path.exists():
            return ValidationResult(passed=False, reasons=[f"Analysis file missing: {path}"])

        data = json.loads(path.read_text())

        bpm = data.get("bpm", 0)
        if not (60 <= bpm <= 200):
            reasons.append(f"BPM out of range: {bpm} (need 60-200)")

        key = data.get("key", "")
        if not key:
            reasons.append("Key is empty")

        structure = data.get("structure", [])
        if len(structure) < 3:
            reasons.append(f"Too few segments: {len(structure)} (need >= 3)")

        chords = data.get("chords", [])
        if not chords:
            reasons.append("Chords are empty")

        duration = data.get("duration", 0)
        if duration <= 90:
            reasons.append(f"Duration too short: {duration}s (need > 90s)")

        energy = data.get("energy_curve", [])
        if not energy:
            reasons.append("Energy curve is empty")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_stems(self, track_id: str, **kwargs) -> ValidationResult:
        import numpy as np
        import soundfile as sf

        reasons = []
        stems_dir = self._root / "stems" / track_id
        stem_names = ("vocals", "drums", "bass", "other")

        # Check all 4 stems exist
        for stem in stem_names:
            if not (stems_dir / f"{stem}.wav").exists():
                reasons.append(f"Missing stem: {stem}.wav")

        if reasons:
            return ValidationResult(passed=False, reasons=reasons)

        # Get source duration for comparison
        source_path = self._root / "raw" / f"{track_id}.mp3"
        if source_path.exists():
            import librosa
            source_duration = librosa.get_duration(path=str(source_path))
        else:
            source_duration = None

        for stem in stem_names:
            path = stems_dir / f"{stem}.wav"
            data, sr = sf.read(str(path))
            duration = len(data) / sr

            # Duration within 3s of source
            if source_duration is not None and abs(duration - source_duration) > 3.0:
                reasons.append(
                    f"{stem}: duration {duration:.1f}s too far from source {source_duration:.1f}s (max 3s)"
                )

            rms_linear = np.sqrt(np.mean(data ** 2))
            if rms_linear > 0:
                rms_db = 20 * np.log10(rms_linear)
            else:
                rms_db = -100.0

            if rms_db < -40:
                reasons.append(f"{stem}: RMS {rms_db:.1f} dBFS below -40 dBFS threshold")

            if stem == "vocals" and rms_db < -30:
                reasons.append(f"vocals: RMS {rms_db:.1f} dBFS below -30 dBFS vocal threshold")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_process_vocals(self, track_id: str, **kwargs) -> ValidationResult:
        import numpy as np
        import soundfile as sf

        reasons = []
        stems_dir = self._root / "stems" / track_id

        processed_path = stems_dir / "vocals_processed.wav"
        hook_path = stems_dir / "vocals_hook.wav"

        if not processed_path.exists():
            reasons.append(f"Missing vocals_processed.wav")
        if not hook_path.exists():
            reasons.append(f"Missing vocals_hook.wav")

        if reasons:
            return ValidationResult(passed=False, reasons=reasons)

        # Check processed vocal RMS is between -18 and -12 dBFS
        data, sr = sf.read(str(processed_path))
        rms_linear = np.sqrt(np.mean(data ** 2))
        if rms_linear > 0:
            rms_db = 20 * np.log10(rms_linear)
        else:
            rms_db = -100.0

        if rms_db < -18:
            reasons.append(f"Processed vocal RMS {rms_db:.1f} dBFS below -18 dBFS")
        elif rms_db > -12:
            reasons.append(f"Processed vocal RMS {rms_db:.1f} dBFS above -12 dBFS")

        # Check hook duration > 10s
        hook_data, hook_sr = sf.read(str(hook_path))
        hook_duration = len(hook_data) / hook_sr
        if hook_duration < 10.0:
            reasons.append(f"Hook too short: {hook_duration:.1f}s (need > 10s)")

        # Check no significant energy below 80Hz
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data
        fft = np.fft.rfft(mono)
        freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
        sub_80_energy = np.sum(np.abs(fft[freqs < 80]) ** 2)
        total_energy = np.sum(np.abs(fft) ** 2)
        sub_80_ratio = sub_80_energy / total_energy if total_energy > 0 else 0
        if sub_80_ratio > 0.01:
            reasons.append(f"Sub-80Hz energy ratio {sub_80_ratio:.4f} exceeds 1% threshold")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_generate_instrumentation(self, track_id: str, **kwargs) -> ValidationResult:
        import numpy as np
        import soundfile as sf

        reasons = []
        wav_path = self._root / "generated" / f"{track_id}_instrumental.wav"

        if not wav_path.exists():
            return ValidationResult(passed=False, reasons=[f"Missing instrumental WAV: {wav_path}"])

        data, sr = sf.read(str(wav_path))
        duration = len(data) / sr

        if duration < 180:
            reasons.append(f"Duration {duration:.1f}s < 180s minimum")

        # BPM check — caller passes measured_bpm from analysis
        target_bpm = kwargs.get("measured_bpm")
        if target_bpm is not None:
            analysis_path = self._root / "analysis" / f"{track_id}_analysis.json"
            if analysis_path.exists():
                analysis = json.loads(analysis_path.read_text())
                expected_bpm = analysis.get("bpm", 0)
                if abs(target_bpm - expected_bpm) > 2:
                    reasons.append(
                        f"BPM {target_bpm:.1f} deviates > ±2 from target {expected_bpm:.1f}"
                    )

        # Vocal contamination check — caller passes vocal_rms_db from Demucs
        vocal_rms_db = kwargs.get("vocal_rms_db")
        if vocal_rms_db is not None and vocal_rms_db > -30:
            reasons.append(f"Vocal contamination: vocal stem RMS {vocal_rms_db:.1f} dBFS > -30 dBFS")

        # Peak check
        peak = np.max(np.abs(data))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
        if peak_db > -0.5:
            reasons.append(f"Peak {peak_db:.1f} dBFS exceeds -0.5 dBFS limit")

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    def _validate_arrange_remix(self, track_id: str, **kwargs) -> ValidationResult:
        import numpy as np
        import soundfile as sf

        reasons = []
        mix_path = self._root / "mixes" / f"{track_id}_rough.wav"

        if not mix_path.exists():
            return ValidationResult(passed=False, reasons=["Missing rough mix WAV"])

        info = sf.info(str(mix_path))

        if info.subtype != "PCM_24":
            reasons.append(f"Not 24-bit: {info.subtype}")

        if info.channels != 2:
            reasons.append(f"Not stereo: {info.channels} channels")

        data, sr = sf.read(str(mix_path))
        duration = len(data) / sr

        # Duration matches instrumental ±500ms
        instrumental_duration = kwargs.get("instrumental_duration")
        if instrumental_duration is not None:
            if abs(duration - instrumental_duration) > 0.5:
                reasons.append(
                    f"Duration {duration:.1f}s differs from instrumental "
                    f"{instrumental_duration:.1f}s by > 500ms"
                )

        # Peak -3 to -0.5 dBFS
        peak = np.max(np.abs(data))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100.0
        if peak_db < -3.5:
            reasons.append(f"Peak {peak_db:.1f} dBFS below -3.5 dBFS")
        elif peak_db > -0.5:
            reasons.append(f"Peak {peak_db:.1f} dBFS above -0.5 dBFS")

        # Hook audible at drop position
        analysis_path = self._root / "analysis" / f"{track_id}_analysis.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text())
            bpm = analysis.get("bpm", 126.0)
            drop_pos = 33 * 4 * (60.0 / bpm)
            drop_sample = int(drop_pos * sr)
            window_end = min(drop_sample + sr, len(data))
            if drop_sample < len(data):
                window = data[drop_sample:window_end]
                if window.ndim > 1:
                    window = window.mean(axis=1)
                rms = np.sqrt(np.mean(window ** 2))
                rms_db = 20 * np.log10(rms) if rms > 0 else -100.0
                if rms_db < -30:
                    reasons.append(
                        f"Hook not audible at drop ({drop_pos:.1f}s): "
                        f"RMS {rms_db:.1f} dBFS"
                    )

        return ValidationResult(passed=len(reasons) == 0, reasons=reasons)

    _validators = {
        "scrape": _validate_scrape,
        "download": _validate_download,
        "analyze": _validate_analyze,
        "stems": _validate_stems,
        "process_vocals": _validate_process_vocals,
        "generate_instrumentation": _validate_generate_instrumentation,
        "arrange_remix": _validate_arrange_remix,
    }
