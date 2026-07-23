import re

import yaml


class ConfigLoader(yaml.FullLoader):
    """YAML loader used by pymss-core model configuration files."""

    pass


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^[-+]?(
            ([0-9][0-9_]*)?\.[0-9_]+([eE][-+]?[0-9]+)?
            |[0-9][0-9_]*[eE][-+]?[0-9]+
            |\.(inf|Inf|INF)
            |\.(nan|NaN|NAN)
        )$""",
        re.X,
    ),
    list("-+0123456789."),
)


class AttrDict(dict):
    """Dictionary that recursively exposes keys as attributes.

    Args:
        data (Mapping | None, optional): Data value. Defaults to None.
        **kwargs: Additional keyword arguments.

    Example:
        >>> cfg = AttrDict({"audio": {"chunk_size": 485100}})
        >>> cfg.audio.chunk_size
        485100"""

    def __init__(self, data=None, **kwargs):
        """Initialize the instance.

        Args:
            data (Mapping | None, optional): Data value. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            None: This method completes for its side effects."""
        super().__init__()
        for key, value in dict(data or {}, **kwargs).items():
            self[key] = value

    def __getattr__(self, key):
        """Return a missing attribute from the underlying mapping.

        Args:
            key (str): Key value.

        Returns:
            Any: Computed result."""
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        """Store an attribute assignment in the underlying mapping.

        Args:
            key (str): Key value.
            value (Any): Value value.

        Returns:
            None: This method completes for its side effects."""
        self[key] = value

    def __setitem__(self, key, value):
        """Store an item after recursively converting nested dictionaries.

        Args:
            key (str): Key value.
            value (Any): Value value.

        Returns:
            None: This method completes for its side effects."""
        super().__setitem__(key, to_attrdict(value))


def to_attrdict(value):
    """Recursively convert dictionaries to AttrDict objects.

    Args:
        value (Any): Value value.

    Returns:
        Any: Converted value with nested dictionaries wrapped as AttrDict."""
    if isinstance(value, dict):
        return value if isinstance(value, AttrDict) else AttrDict(value)
    if isinstance(value, list):
        return [to_attrdict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_attrdict(item) for item in value)
    return value


def to_plain(value):
    """Recursively convert AttrDict objects back to plain Python containers.

    Args:
        value (Any): Value value.

    Returns:
        Any: Converted value using plain dictionaries, lists, and tuples."""
    if isinstance(value, AttrDict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain(item) for item in value)
    return value


def load_config(path):
    """Load a YAML model configuration file.

    Args:
        path (str | os.PathLike): File system path.

    Returns:
        AttrDict: Parsed configuration with attribute access.

    Example:
        >>> config = load_config("config.yaml")
        >>> config.inference.batch_size"""
    with open(path, encoding="utf-8") as f:
        return to_attrdict(yaml.load(f, Loader=ConfigLoader))
