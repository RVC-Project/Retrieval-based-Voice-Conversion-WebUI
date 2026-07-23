from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


WORKFLOW_TEMPLATE = """version: 1

defaults:
  device: auto
  output_format: wav
  model_dir: null
  inference_params:
    normalize: false

steps:
  - id: split
    model: bs_roformer_voc_hyperacev2
    input: input
    stems: [vocals, other]
    inference_params:
      overlap_size: 48000
    save:
      vocals: vocal
      other: other

  - id: dereverb
    model: UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband
    input: split.other
    stems: [Dry]
    inference_params:
      overlap_size: 22050
    save:
      Dry: dry

  - id: harmony
    model: your_harmony_model
    input: dereverb.Dry
    stems: [other]
    inference_params:
      overlap_size: 22050
    save:
      other: harmony_other
"""


_STEP_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_OUTPUT_FORMATS = {"wav", "flac", "mp3", "m4a"}
_OUTPUT_LAYOUTS = {"folders", "flat"}
_DEFAULT_AUDIO_PARAMS = {
    "wav_bit_depth": "FLOAT",
    "flac_bit_depth": "PCM_24",
    "mp3_bit_rate": "320k",
    "m4a_bit_rate": "512k",
    "m4a_codec": "aac",
    "m4a_aac_at_quality": 2,
}


class WorkflowError(ValueError):
    """Raised when a workflow definition or run is invalid."""


@dataclass(frozen=True)
class WorkflowStep:
    id: str
    model: str | None = None
    input: str = "input"
    stems: list[str] | None = None
    save: dict[str, Any] = field(default_factory=dict)
    model_type: str | None = None
    model_path: str | None = None
    config_path: str | None = None
    device: str | None = None
    model_dir: str | None = None
    output_format: str | None = None
    inference_params: dict[str, Any] = field(default_factory=dict)
    use_tta: bool | None = None


@dataclass(frozen=True)
class Workflow:
    version: int
    defaults: dict[str, Any]
    steps: list[WorkflowStep]


@dataclass(frozen=True)
class AudioArtifact:
    audio: np.ndarray
    sample_rate: int


@dataclass
class WorkflowTrackState:
    path: str
    track_name: str
    artifacts: dict[str, AudioArtifact] = field(default_factory=dict)
    active: bool = True


def load_workflow_file(path: str | os.PathLike) -> Workflow:
    """Load a workflow YAML/JSON file."""
    workflow_path = Path(path)
    try:
        data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise WorkflowError(f"Invalid workflow YAML: {exc}") from exc
    except OSError as exc:
        raise WorkflowError(f"Cannot read workflow file: {workflow_path}") from exc
    return load_workflow_data(data)


def load_workflow_data(data: Any) -> Workflow:
    """Parse workflow data from a Python mapping."""
    if not isinstance(data, dict):
        raise WorkflowError("Workflow file must contain a mapping.")
    version = data.get("version")
    if version != 1:
        raise WorkflowError("workflow version must be 1.")
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise WorkflowError("defaults must be a mapping.")
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkflowError("steps must be a non-empty list.")
    steps = [_parse_step(index, item) for index, item in enumerate(raw_steps, start=1)]
    workflow = Workflow(version=int(version), defaults=dict(defaults), steps=steps)
    validate_workflow_structure(workflow)
    return workflow


def write_workflow_template(path: str | os.PathLike, *, overwrite: bool = False) -> Path:
    """Write a starter workflow YAML file."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise WorkflowError(f"Workflow file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(WORKFLOW_TEMPLATE, encoding="utf-8")
    return output_path


def validate_workflow(
    workflow: Workflow,
    *,
    model_dir: str | os.PathLike | None = None,
    require_model_files: bool = False,
    model_resolver: Callable[..., Any] | None = None,
) -> Workflow:
    """Validate workflow references and optionally model catalog entries."""
    validate_workflow_structure(workflow)
    _validate_step_references(workflow)
    if model_resolver is not None or require_model_files:
        resolver = model_resolver or _default_model_resolver
        for step in workflow.steps:
            if step.model_path:
                _validate_explicit_model_files(step, require_model_files=require_model_files)
                continue
            step_model_dir = _step_option(workflow, step, "model_dir", model_dir)
            resolver(
                step.model,
                model_dir=step_model_dir,
                require_supported=True,
                require_exists=require_model_files,
            )
    return workflow


def validate_workflow_structure(workflow: Workflow) -> Workflow:
    """Validate syntax that does not require cross-step analysis."""
    seen = set()
    for step in workflow.steps:
        if not _STEP_ID_RE.match(step.id):
            raise WorkflowError(f"Invalid step id {step.id!r}.")
        if step.id in seen:
            raise WorkflowError(f"Duplicate step id: {step.id}")
        seen.add(step.id)
        if bool(step.model) == bool(step.model_path):
            raise WorkflowError(f"step {step.id!r} requires exactly one of model or model_path.")
        if step.model_path and not step.model_type:
            raise WorkflowError(f"step {step.id!r} requires model_type when model_path is used.")
        if step.stems is not None and not step.stems:
            raise WorkflowError(f"step {step.id!r} stems must not be empty.")
        if step.output_format is not None and str(step.output_format).lower() not in _OUTPUT_FORMATS:
            raise WorkflowError(f"step {step.id!r} has unsupported output_format {step.output_format!r}.")
    default_format = workflow.defaults.get("output_format")
    if default_format is not None and str(default_format).lower() not in _OUTPUT_FORMATS:
        raise WorkflowError(f"defaults.output_format must be one of: {sorted(_OUTPUT_FORMATS)}.")
    default_inference_params = workflow.defaults.get("inference_params")
    if default_inference_params is not None and not isinstance(default_inference_params, dict):
        raise WorkflowError("defaults.inference_params must be a mapping.")
    return workflow


class WorkflowRunner:
    """Run a parsed pymss workflow over one file or a direct folder."""

    def __init__(
        self,
        workflow: Workflow,
        *,
        model_dir: str | os.PathLike | None = None,
        device: str | None = None,
        output_format: str | None = None,
        download: bool = False,
        source: str = "modelscope",
        endpoint: str | None = None,
        audio_params: dict[str, Any] | None = None,
        logger: Any = None,
        debug: bool = False,
        separator_factory: Callable[..., Any] | None = None,
        audio_loader: Callable[..., Any] | None = None,
        audio_saver: Callable[..., Any] | None = None,
        continue_on_error: bool = False,
        output_layout: str = "folders",
    ):
        self.workflow = validate_workflow(workflow)
        self.model_dir = model_dir
        self.device = device
        self.output_format = output_format
        self.download = bool(download)
        self.source = source
        self.endpoint = endpoint
        self.audio_params = {**_DEFAULT_AUDIO_PARAMS, **(audio_params or {})}
        self.logger = logger
        self.debug = bool(debug)
        self.separator_factory = separator_factory or _default_separator_factory
        self.audio_loader = audio_loader or _default_audio_loader
        self.audio_saver = audio_saver or _default_audio_saver
        self.continue_on_error = bool(continue_on_error)
        self.output_layout = _validate_output_layout(output_layout)

    def run(self, input_path: str | os.PathLike, output_dir: str | os.PathLike) -> list[str]:
        """Run the workflow and return successfully processed basenames."""
        paths = _input_files(input_path)
        output_root = Path(output_dir)
        tracks = [
            track
            for path, track_name in zip(paths, _unique_track_names(paths))
            for track in [self._load_track(path, track_name)]
            if track is not None
        ]
        for step in self.workflow.steps:
            active_tracks = [track for track in tracks if track.active]
            if not active_tracks:
                break
            try:
                with self._open_separator(step) as separator:
                    for track in active_tracks:
                        self._run_step_for_track(step, separator, track, output_root)
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                for track in active_tracks:
                    self._mark_track_failed(track, exc)
        return [os.path.basename(track.path) for track in tracks if track.active]

    def _load_track(self, path: str, track_name: str) -> WorkflowTrackState | None:
        try:
            mix, sr = self.audio_loader(path, sr=None, mono=False)
            return WorkflowTrackState(
                path=path,
                track_name=track_name,
                artifacts={"input": AudioArtifact(_to_model_audio(mix), int(sr))},
            )
        except Exception as exc:
            if self.continue_on_error and self.logger is not None:
                self.logger.warning("Cannot process workflow track %s: %s", path, exc)
                return None
            raise

    def _run_step_for_track(
        self,
        step: WorkflowStep,
        separator: Any,
        track: WorkflowTrackState,
        output_root: Path,
    ) -> None:
        try:
            artifact = _resolve_input_artifact(track.artifacts, step.input)
            sample_rate = int(separator.config.audio.get("sample_rate", artifact.sample_rate))
            model_audio = _ensure_sample_rate(_to_model_audio(artifact.audio), artifact.sample_rate, sample_rate)
            stems = _requested_stems(step)
            if getattr(separator, "model_type", None) == "vr":
                results = separator.separate(model_audio, pbar=False)
            else:
                results = separator.separate(model_audio, pbar=False, stems=stems)
            selected = _select_results(step, results)
            for stem, audio in selected.items():
                track.artifacts[f"{step.id}.{stem}"] = AudioArtifact(_to_model_audio(audio), sample_rate)
            self._save_results(step, selected, sample_rate, output_root, track.track_name)
            del selected, results
        except Exception as exc:
            if not self.continue_on_error:
                raise
            self._mark_track_failed(track, exc)

    def _mark_track_failed(self, track: WorkflowTrackState, exc: Exception) -> None:
        track.active = False
        if self.logger is not None:
            self.logger.warning("Cannot process workflow track %s: %s", track.path, exc)

    def _open_separator(self, step: WorkflowStep):
        if self.download and step.model:
            from .model_download import download_model

            download_model(
                step.model,
                model_dir=_step_option(self.workflow, step, "model_dir", self.model_dir),
                source=self.source,
                endpoint=self.endpoint,
            )
        separator_kwargs = {
            "model_dir": _step_option(self.workflow, step, "model_dir", self.model_dir),
            "device": _step_option(self.workflow, step, "device", self.device),
            "output_format": _step_option(self.workflow, step, "output_format", self.output_format) or "wav",
            "audio_params": self.audio_params,
            "use_tta": bool(_step_option(self.workflow, step, "use_tta", None) or False),
            "logger": self.logger,
            "debug": self.debug,
            "inference_params": _merged_inference_params(self.workflow, step),
        }
        model_name = step.model
        if step.model_path:
            model_name = Path(step.model_path).stem
            separator_kwargs.update(
                {
                    "model_type": step.model_type,
                    "model_path": step.model_path,
                    "config_path": step.config_path,
                }
            )
        separator = self.separator_factory(
            model_name,
            **separator_kwargs,
        )
        return _SeparatorContext(separator)

    def _save_results(
        self,
        step: WorkflowStep,
        results: dict[str, np.ndarray],
        sample_rate: int,
        output_root: Path,
        track_name: str,
    ) -> None:
        output_format = str(_step_option(self.workflow, step, "output_format", self.output_format) or "wav").lower()
        for stem, audio in results.items():
            save_dirs = _save_dirs(step, stem)
            for save_dir in save_dirs:
                target_dir = output_root / save_dir
                if self.output_layout == "folders":
                    target_dir = output_root / track_name / save_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                safe_stem = _safe_filename_part(stem)
                target = target_dir / f"{track_name}_{safe_stem}.{output_format}"
                self.audio_saver(str(target), _to_save_audio(audio), sample_rate, output_format, self.audio_params)


class _SeparatorContext:
    def __init__(self, separator):
        self.separator = separator

    def __enter__(self):
        enter = getattr(self.separator, "__enter__", None)
        return enter() if enter is not None else self.separator

    def __exit__(self, exc_type, exc_value, traceback):
        exit_method = getattr(self.separator, "__exit__", None)
        if exit_method is not None:
            return exit_method(exc_type, exc_value, traceback)
        close = getattr(self.separator, "close", None)
        if close is not None:
            close()
        return False


def run_workflow_file(
    config_path: str | os.PathLike,
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    **runner_kwargs,
) -> list[str]:
    """Load and run a workflow file."""
    workflow = load_workflow_file(config_path)
    return WorkflowRunner(workflow, **runner_kwargs).run(input_path, output_dir)


def _validate_output_layout(value: str) -> str:
    layout = str(value).strip().lower()
    if layout not in _OUTPUT_LAYOUTS:
        raise WorkflowError(f"output_layout must be one of: {sorted(_OUTPUT_LAYOUTS)}.")
    return layout


def _parse_step(index: int, data: Any) -> WorkflowStep:
    if not isinstance(data, dict):
        raise WorkflowError(f"step #{index} must be a mapping.")
    step_id = data.get("id")
    if not isinstance(step_id, str) or not step_id.strip():
        raise WorkflowError(f"step #{index} requires a non-empty id.")
    model = _parse_optional_string(data.get("model"))
    model_path = _parse_optional_string(data.get("model_path"))
    return WorkflowStep(
        id=step_id.strip(),
        model=model,
        input=_parse_input_value(data.get("input", "input"), step_id),
        stems=_parse_stems(data.get("stems"), step_id),
        save=_parse_save(data.get("save"), step_id),
        model_type=_parse_optional_string(data.get("model_type")),
        model_path=model_path,
        config_path=_parse_optional_string(data.get("config_path")),
        device=_parse_optional_string(data.get("device")),
        model_dir=_parse_optional_string(data.get("model_dir")),
        output_format=_parse_optional_string(data.get("output_format")),
        inference_params=_parse_mapping(data.get("inference_params"), step_id, "inference_params"),
        use_tta=_parse_optional_bool(data.get("use_tta"), step_id, "use_tta"),
    )


def _parse_input_value(value: Any, step_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"step {step_id!r} input must be a non-empty string.")
    return value.strip()


def _parse_stems(value: Any, step_id: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stems = [value]
    elif isinstance(value, list):
        stems = value
    else:
        raise WorkflowError(f"step {step_id!r} stems must be a string or list.")
    result = [str(item).strip() for item in stems if str(item).strip()]
    if not result:
        raise WorkflowError(f"step {step_id!r} stems must not be empty.")
    return result


def _parse_save(value: Any, step_id: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise WorkflowError(f"step {step_id!r} save must be a mapping.")
    result = {}
    for stem, target in value.items():
        stem_name = str(stem).strip()
        if not stem_name:
            raise WorkflowError(f"step {step_id!r} save contains an empty stem name.")
        result[stem_name] = target
    return result


def _parse_mapping(value: Any, step_id: str, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise WorkflowError(f"step {step_id!r} {field_name} must be a mapping.")
    return dict(value)


def _parse_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _parse_optional_bool(value: Any, step_id: str, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise WorkflowError(f"step {step_id!r} {field_name} must be a boolean.")


def _validate_step_references(workflow: Workflow) -> None:
    seen = {"input"}
    required_outputs: dict[str, set[str]] = {step.id: set() for step in workflow.steps}
    for step in workflow.steps:
        if step.input != "input":
            ref_step, ref_stem = _split_artifact_ref(step.input, step.id)
            if ref_step not in seen:
                raise WorkflowError(f"step {step.id!r} input references unknown step: {ref_step}")
            required_outputs.setdefault(ref_step, set()).add(ref_stem)
        for stem in step.save:
            required_outputs[step.id].add(stem)
        seen.add(step.id)

    for step in workflow.steps:
        if step.stems is None:
            continue
        requested = {stem.lower() for stem in step.stems}
        for stem in required_outputs.get(step.id, set()):
            if stem.lower() not in requested:
                raise WorkflowError(
                    f"step {step.id!r} must request {step.id}.{stem}; add {stem!r} to stems or omit stems."
                )


def _split_artifact_ref(value: str, current_step_id: str) -> tuple[str, str]:
    if "." not in value:
        raise WorkflowError(f"step {current_step_id!r} input must be 'input' or '<step>.<stem>'.")
    step_id, stem = value.split(".", 1)
    step_id = step_id.strip()
    stem = stem.strip()
    if not step_id or not stem:
        raise WorkflowError(f"step {current_step_id!r} input must be 'input' or '<step>.<stem>'.")
    return step_id, stem


def _validate_explicit_model_files(step: WorkflowStep, *, require_model_files: bool) -> None:
    if not require_model_files:
        return
    missing = []
    if step.model_path and not Path(step.model_path).is_file():
        missing.append(step.model_path)
    if step.config_path and not Path(step.config_path).is_file():
        missing.append(step.config_path)
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))


def _resolve_input_artifact(artifacts: dict[str, AudioArtifact], ref: str) -> AudioArtifact:
    if ref == "input":
        return artifacts["input"]
    if ref in artifacts:
        return artifacts[ref]
    ref_step, ref_stem = ref.split(".", 1)
    for key, artifact in artifacts.items():
        if not key.startswith(f"{ref_step}."):
            continue
        _, stem = key.split(".", 1)
        if stem.lower() == ref_stem.lower():
            return artifact
    raise WorkflowError(f"Missing workflow input artifact: {ref}")


def _requested_stems(step: WorkflowStep) -> list[str] | None:
    if step.stems is not None:
        return list(step.stems)
    if step.save:
        return list(step.save)
    return None


def _select_results(step: WorkflowStep, results: dict[str, Any]) -> dict[str, np.ndarray]:
    requested = _requested_stems(step)
    if requested is None:
        requested = list(results)
    selected = {}
    for stem in requested:
        actual = _find_stem(results, stem)
        selected[actual] = np.asarray(results[actual], dtype=np.float32)
    return selected


def _find_stem(results: dict[str, Any], stem: str) -> str:
    if stem in results:
        return stem
    lower = str(stem).lower()
    for key in results:
        if str(key).lower() == lower:
            return key
    raise WorkflowError(f"Model did not return requested stem {stem!r}. Available stems: {list(results)}")


def _save_dirs(step: WorkflowStep, stem: str) -> list[str]:
    if not step.save:
        return []
    target = _case_insensitive_get(step.save, stem)
    if target in (None, False, ""):
        return []
    if target is True:
        return [step.id]
    if isinstance(target, list):
        return [str(item).strip() for item in target if str(item).strip()]
    target = str(target).strip()
    return [target] if target else []


def _case_insensitive_get(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    lower = str(key).lower()
    for item_key, value in mapping.items():
        if str(item_key).lower() == lower:
            return value
    return None


def _to_model_audio(audio: Any) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise WorkflowError(f"Expected mono or stereo audio, got shape {array.shape}.")
    if array.shape[0] in (1, 2):
        return np.ascontiguousarray(array)
    if array.shape[1] in (1, 2):
        return np.ascontiguousarray(array.T)
    raise WorkflowError(f"Expected mono or stereo audio, got shape {array.shape}.")


def _to_save_audio(audio: Any) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim != 2:
        raise WorkflowError(f"Expected mono or stereo audio, got shape {array.shape}.")
    if array.shape[1] in (1, 2):
        return np.ascontiguousarray(array)
    if array.shape[0] in (1, 2):
        return np.ascontiguousarray(array.T)
    raise WorkflowError(f"Expected mono or stereo audio, got shape {array.shape}.")


def _ensure_sample_rate(audio: np.ndarray, current_sr: int, target_sr: int) -> np.ndarray:
    if int(current_sr) == int(target_sr):
        return audio
    import librosa

    return np.ascontiguousarray(
        librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(current_sr), target_sr=int(target_sr), axis=-1)
    )


def _input_files(input_path: str | os.PathLike) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if path.is_dir():
        return [str(item) for item in sorted(path.iterdir()) if item.is_file()]
    raise WorkflowError(f"Input path does not exist: {path}")


def _unique_track_names(paths: list[str]) -> list[str]:
    original_stems = {Path(path).stem for path in paths}
    next_suffix: dict[str, int] = {}
    used: set[str] = set()
    names = []
    for path in paths:
        stem = Path(path).stem
        if stem not in used:
            used.add(stem)
            names.append(stem)
            continue
        suffix = next_suffix.get(stem, 2)
        candidate = f"{stem}_{suffix}"
        while candidate in used or candidate in original_stems:
            suffix += 1
            candidate = f"{stem}_{suffix}"
        next_suffix[stem] = suffix + 1
        used.add(candidate)
        names.append(candidate)
    return names


def _step_option(workflow: Workflow, step: WorkflowStep, key: str, override: Any = None) -> Any:
    value = getattr(step, key, None)
    if value is not None:
        return value
    if override is not None:
        return override
    return workflow.defaults.get(key)


def _merged_inference_params(workflow: Workflow, step: WorkflowStep) -> dict[str, Any]:
    defaults = workflow.defaults.get("inference_params") or {}
    return {**defaults, **(step.inference_params or {})}


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[\\/:\0]+", "_", str(value)).strip()
    return safe or "stem"


def _default_separator_factory(model_name: str, **kwargs):
    model_type = kwargs.pop("model_type", None)
    model_path = kwargs.pop("model_path", None)
    config_path = kwargs.pop("config_path", None)
    if model_path:
        kwargs.pop("model_dir", None)
        from .separator import MSSeparator

        return MSSeparator(model_type=model_type, model_path=model_path, config_path=config_path, **kwargs)

    from .model_registry import create_separator
    return create_separator(model_name, **kwargs)


def _default_model_resolver(*args, **kwargs):
    from .model_registry import resolve_model

    return resolve_model(*args, **kwargs)


def _default_audio_loader(*args, **kwargs):
    from .audio_io import load_audio

    return load_audio(*args, **kwargs)


def _default_audio_saver(*args, **kwargs):
    from .audio_io import save_audio

    return save_audio(*args, **kwargs)
