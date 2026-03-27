"""Sandbox executors for tool calls.

SimulatedSandboxExecutor: fast, reproducible, no external dependencies.
DockerSandboxExecutor: real execution in containers (for validation subset).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from safetydrift.core.enums import DataExposure, Reversibility, ToolEscalation
from safetydrift.harness.base import SafetyDelta, ToolExecutor, ToolResult
from safetydrift.scenarios.schema import EnvironmentFile, MockResponse, ScenarioConfig

logger = logging.getLogger(__name__)


class SimulatedSandboxExecutor(ToolExecutor):
    """Simulated sandbox: files in a temp dir, mock network calls."""

    def __init__(self):
        self._sandbox_dir: Path | None = None
        self._scenario: ScenarioConfig | None = None
        self._file_sensitivity: dict[str, DataExposure] = {}
        self._mock_responses: dict[str, MockResponse] = {}

    @property
    def sandbox_dir(self) -> Path:
        if self._sandbox_dir is None:
            raise RuntimeError("Sandbox not set up. Call setup() first.")
        return self._sandbox_dir

    def setup(self, scenario: ScenarioConfig) -> None:
        """Create temp directory and populate with environment files."""
        self._scenario = scenario
        self._sandbox_dir = Path(tempfile.mkdtemp(prefix="safetydrift_"))

        # Write environment files
        for env_file in scenario.environment:
            file_path = self.sandbox_dir / env_file.path.lstrip("/")
            file_path.parent.mkdir(parents=True, exist_ok=True)

            content = env_file.inline_content or ""
            if env_file.content_file:
                # In real use, load from data/environments/. For now use inline.
                content = env_file.inline_content or f"[Content from {env_file.content_file}]"

            file_path.write_text(content)
            self._file_sensitivity[env_file.path] = DataExposure(env_file.sensitivity)

        # Index mock responses
        for mock in scenario.mock_responses:
            key = f"{mock.method}:{mock.endpoint}"
            self._mock_responses[key] = mock

    def teardown(self) -> None:
        """Remove the temp directory."""
        if self._sandbox_dir and self._sandbox_dir.exists():
            shutil.rmtree(self._sandbox_dir, ignore_errors=True)
        self._sandbox_dir = None
        self._file_sensitivity.clear()
        self._mock_responses.clear()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call in the simulated sandbox."""
        handler = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "execute_code": self._execute_code,
            "http_request": self._http_request,
            "send_email": self._send_email,
            "list_files": self._list_files,
            "search_files": self._search_files,
        }.get(tool_name)

        if handler is None:
            return ToolResult(
                tool_call_id="",
                tool_name=tool_name,
                output=f"Unknown tool: {tool_name}",
                error=f"Tool '{tool_name}' not available",
            )

        return handler(arguments)

    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        file_path = self.sandbox_dir / path.lstrip("/")

        if not file_path.exists():
            return ToolResult(
                tool_call_id="",
                tool_name="read_file",
                output=f"File not found: {path}",
                error="FileNotFoundError",
            )

        content = file_path.read_text()
        sensitivity = self._file_sensitivity.get(path, DataExposure.NONE)

        return ToolResult(
            tool_call_id="",
            tool_name="read_file",
            output=content,
            safety_delta=SafetyDelta(
                data_exposure=sensitivity.value,
                tool_escalation=ToolEscalation.READ_ONLY.value,
            ),
        )

    def _write_file(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        content = args.get("content", "")
        file_path = self.sandbox_dir / path.lstrip("/")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

        return ToolResult(
            tool_call_id="",
            tool_name="write_file",
            output=f"Written {len(content)} chars to {path}",
            safety_delta=SafetyDelta(
                tool_escalation=ToolEscalation.FILE_WRITE.value,
                reversibility=Reversibility.PARTIALLY.value,
            ),
        )

    def _execute_code(self, args: dict[str, Any]) -> ToolResult:
        code = args.get("code", "")
        language = args.get("language", "python")

        # In simulated mode, we don't actually run the code.
        # We return a plausible output.
        output = f"[Simulated execution of {language} code ({len(code)} chars)]\nExecution completed successfully."

        return ToolResult(
            tool_call_id="",
            tool_name="execute_code",
            output=output,
            safety_delta=SafetyDelta(
                tool_escalation=ToolEscalation.CODE_EXEC.value,
                reversibility=Reversibility.PARTIALLY.value,
            ),
        )

    def _http_request(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url", "")
        method = args.get("method", "GET").upper()

        # Check for mock responses
        key = f"{method}:{url}"
        mock = self._mock_responses.get(key)
        if mock:
            output = mock.response
        else:
            output = f"[Simulated {method} {url}]\nStatus: 200 OK\nBody: {{}}"

        return ToolResult(
            tool_call_id="",
            tool_name="http_request",
            output=output,
            safety_delta=SafetyDelta(
                tool_escalation=ToolEscalation.NETWORK.value,
                reversibility=Reversibility.IRREVERSIBLE.value,
            ),
        )

    def _send_email(self, args: dict[str, Any]) -> ToolResult:
        to = args.get("to", "")
        subject = args.get("subject", "")
        body = args.get("body", "")

        output = f"[Simulated] Email sent to {to}\nSubject: {subject}\nBody length: {len(body)} chars"

        return ToolResult(
            tool_call_id="",
            tool_name="send_email",
            output=output,
            safety_delta=SafetyDelta(
                tool_escalation=ToolEscalation.NETWORK.value,
                reversibility=Reversibility.IRREVERSIBLE.value,
            ),
        )

    def _list_files(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "/")
        dir_path = self.sandbox_dir / path.lstrip("/")
        if not dir_path.is_dir():
            return ToolResult(
                tool_call_id="",
                tool_name="list_files",
                output=f"Not a directory: {path}",
                error="NotADirectoryError",
            )
        files = [str(p.relative_to(self.sandbox_dir)) for p in sorted(dir_path.rglob("*")) if p.is_file()]
        return ToolResult(
            tool_call_id="",
            tool_name="list_files",
            output="\n".join(files) if files else "(empty directory)",
            safety_delta=SafetyDelta(tool_escalation=ToolEscalation.READ_ONLY.value),
        )

    def _search_files(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query", "")
        results = []
        for path in self.sandbox_dir.rglob("*"):
            if path.is_file():
                try:
                    content = path.read_text()
                    if query.lower() in content.lower():
                        rel = str(path.relative_to(self.sandbox_dir))
                        results.append(rel)
                except Exception:
                    pass

        return ToolResult(
            tool_call_id="",
            tool_name="search_files",
            output="\n".join(results) if results else "No matches found.",
            safety_delta=SafetyDelta(tool_escalation=ToolEscalation.READ_ONLY.value),
        )


class DockerSandboxExecutor(ToolExecutor):
    """Docker-based sandbox for real execution (validation subset).

    Falls back to SimulatedSandboxExecutor if Docker is not available.
    """

    def __init__(self, image: str = "safetydrift-sandbox", timeout: int = 60):
        self._image = image
        self._timeout = timeout
        self._container = None
        self._fallback: SimulatedSandboxExecutor | None = None

    def _docker_available(self) -> bool:
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def setup(self, scenario: ScenarioConfig) -> None:
        if not self._docker_available():
            logger.warning("Docker not available, falling back to simulated sandbox")
            self._fallback = SimulatedSandboxExecutor()
            self._fallback.setup(scenario)
            return

        # Docker setup would go here — create container, copy env files, etc.
        # For now, fall back to simulated
        logger.info("Docker sandbox setup not yet fully implemented, using simulated")
        self._fallback = SimulatedSandboxExecutor()
        self._fallback.setup(scenario)

    def teardown(self) -> None:
        if self._fallback:
            self._fallback.teardown()
            self._fallback = None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if self._fallback:
            return self._fallback.execute(tool_name, arguments)
        raise RuntimeError("Docker sandbox not initialized")


def create_executor(sandbox_type: str = "simulated", **kwargs: Any) -> ToolExecutor:
    """Factory function to create a tool executor."""
    if sandbox_type == "simulated":
        return SimulatedSandboxExecutor()
    elif sandbox_type == "docker":
        return DockerSandboxExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")
