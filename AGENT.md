# Agent Instructions for RVC WebGUI Fork

This document provides essential guidelines and context for AI agents working on this repository. Adhering to these standards ensures consistency and leverages modern Python capabilities.

## Environment & Runtime

- **Python Version**: Assume **Python 3.13**.
- **Virtual Environment**: Always use the local `.venv` directory for the Python interpreter and dependency management. Avoid using the system Python.
- **Dependency Management**: The project uses `uv`. Refer to `pyproject.toml` and `uv.lock` for dependencies.

## Coding Standards

### Modern Python Features

- **Prefer Modern Syntax**: Always use the latest Python features available in 3.13.
- **Legacy Cleanup**: Actively remove "old" Python idioms when encountered.
  - _Example_: Remove explicit `(object)` inheritance from classes (e.g., change `class MyClass(object):` to `class MyClass:`).
  - _Example_: Use `pathlib` over `os.path` for file operations.

### Type Hinting & Annotations

- **Modern Typing**: Use modern typing features.
  - Prefer `dict[str, int]` over `typing.Dict[str, int]`.
  - Use `list[str]` over `typing.List[str]`.
  - Use the pipe operator `|` for Unions (e.g., `str | None`) instead of `typing.Union` or `typing.Optional`.
- **TypedDict**: Prefer `TypedDict` for defining the structure of dictionaries.
- **Forward References**: Use string literals for forward references or rely on modern behavior (Python 3.13 handles most cases gracefully).
- **Static Analysis**: The codebase aims for high type coverage to support static analysis.

## Repository Context

This repository is a modernized fork of the RVC WebUI. The goal is to improve maintainability, performance, and cross-platform compatibility while preserving core machine learning logic.
