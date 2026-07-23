from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Runtime configuration for the pymss HTTP server."""

    model: str | None = None
    model_dir: str | None = None
    source: str = "modelscope"
    endpoint: str | None = None
    device: str = "auto"
    device_ids: list[int] = field(default_factory=lambda: [0])
    api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    inference_params: dict = field(default_factory=dict)
    max_audio_seconds: float = 600.0
    max_request_bytes: int = 536870912
    max_queue_size: int = 8
    request_timeout_seconds: float = 0.0
    webui: bool = False
