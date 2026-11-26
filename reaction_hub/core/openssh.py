from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SSHRunResult:
    stdout: str
    stderr: str
    returncode: int


class OpenSSHClient:
    """Minimal non-interactive SSH client built on top of the OpenSSH CLI."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str,
        ssh_key_path: str = "",
        timeout: int = 20,
        ssh_binary: str = "ssh",
        force_tty: bool = False,
        batch_mode: bool = True,
        use_clean_bash: bool = False,
        bash_path: str = "bash",
    ):
        self.host = host
        self.port = int(port)
        self.username = username
        self.ssh_key_path = str(ssh_key_path or "").strip()
        self.timeout = int(timeout)
        if ssh_binary == "ssh" and os.name == "nt":
            ssh_binary = r"C:\Windows\System32\OpenSSH\ssh.exe"
        self.ssh_binary = ssh_binary
        self.force_tty = bool(force_tty)
        self.batch_mode = bool(batch_mode)
        self.use_clean_bash = bool(use_clean_bash)
        self.bash_path = str(bash_path or "bash").strip() or "bash"

    def __enter__(self) -> "OpenSSHClient":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        return None

    def _target(self) -> str:
        return f"{self.username}@{self.host}"

    def _mode_summary(self) -> str:
        key_info = self.ssh_key_path if self.ssh_key_path else "<agent/default>"
        return (
            f"target={self._target()} port={self.port} key={key_info} "
            f"tty={self.force_tty} batch_mode={self.batch_mode} "
            f"clean_bash={self.use_clean_bash} bash_path={self.bash_path}"
        )

    @staticmethod
    def _normalize_returncode(returncode: int) -> int:
        # Windows may surface a signed -1 as 4294967295 (uint32 max).
        if returncode >= 2**31:
            return returncode - 2**32
        return returncode

    @staticmethod
    def _strip_connection_closed_line(text: str) -> str:
        lines: list[str] = []
        for line in (text or "").splitlines():
            trimmed = line.strip()
            if trimmed.startswith("Connection to ") and trimmed.endswith(" closed."):
                continue
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _extract_marker(text: str, marker_key: str) -> tuple[str, int | None]:
        pattern = re.compile(rf"{re.escape(marker_key)}=(\d+)")
        found: int | None = None
        kept: list[str] = []
        for line in (text or "").splitlines():
            match = pattern.search(line)
            if match:
                try:
                    found = int(match.group(1))
                except ValueError:
                    found = None
                continue
            kept.append(line)
        return "\n".join(kept), found

    @staticmethod
    def _strip_before_marker(text: str, marker: str) -> str:
        marker_end = (text or "").rfind(marker)
        if marker_end < 0:
            return text or ""
        payload = text[marker_end + len(marker) :]
        if payload.startswith("\r\n"):
            return payload[2:]
        if payload.startswith("\n"):
            return payload[1:]
        return payload

    @staticmethod
    def _trim_text(text: str, *, limit: int = 1200) -> str:
        payload = (text or "").strip()
        if not payload:
            return "<empty>"
        if len(payload) <= limit:
            return payload
        return payload[:limit] + "\n...[truncated]..."

    def _failure_hint(self, detail: str) -> str:
        text = (detail or "").lower()
        if "permission denied (keyboard-interactive)" in text or "permission denied" in text:
            return (
                "Authentication failed in the current runtime context. "
                "Verify the same terminal/environment can run "
                "`ssh -tt user@host \"echo OK\"`, then restart Django/Celery "
                "after setting REACTION_HUB_SSH_FORCE_TTY/REACTION_HUB_SSH_BATCH_MODE."
            )
        if "timed out" in text:
            timeout_hint = "Connection timed out. Check network/VPN reachability to the HPC login node."
            if not self.batch_mode:
                return (
                    f"{timeout_hint} With REACTION_HUB_SSH_BATCH_MODE=False, timeouts can also "
                    "mean SSH is waiting for interactive auth (password/MFA/passphrase) in a "
                    "non-interactive worker process."
                )
            return timeout_hint
        return ""

    def _base_args(self, *, force_tty: bool | None = None) -> list[str]:
        use_tty = self.force_tty if force_tty is None else bool(force_tty)
        batch_text = "yes" if self.batch_mode else "no"
        args = [
            self.ssh_binary,
            "-p",
            str(self.port),
        ]
        if use_tty:
            args.append("-tt")
        args.extend(
            [
            "-o",
            f"BatchMode={batch_text}",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"ConnectTimeout={self.timeout}",
            ]
        )
        if os.name == "nt" and os.environ.get("USERPROFILE"):
            args.extend(
                [
                    "-o",
                    f"UserKnownHostsFile={os.path.join(os.environ['USERPROFILE'], '.ssh', 'known_hosts')}",
                ]
            )
        if self.ssh_key_path:
            args.extend(["-i", self.ssh_key_path])
        return args

    def _wrap_command(self, command: str, *, use_clean_bash: bool | None = None) -> str:
        clean = self.use_clean_bash if use_clean_bash is None else bool(use_clean_bash)
        if not clean:
            return command
        return f"{self.bash_path} --noprofile --norc -c {shlex.quote(command)}"

    def run(
        self,
        command: str,
        *,
        input_text: str | None = None,
        force_tty: bool | None = None,
        use_clean_bash: bool | None = None,
    ) -> SSHRunResult:
        remote_command = self._wrap_command(command, use_clean_bash=use_clean_bash)
        argv = [*self._base_args(force_tty=force_tty), self._target(), remote_command]
        try:
            completed = subprocess.run(
                argv,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
                timeout=max(1, self.timeout),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"SSH binary not found ({exc}): {self.ssh_binary}\nargv={argv}\nSSH context: {self._mode_summary()}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"SSH command timed out after {self.timeout}s\n"
                f"SSH context: {self._mode_summary()}\n"
                f"Command: {command}"
            ) from exc

        return SSHRunResult(
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        )

    def exec_checked(self, command: str) -> str:
        start_marker = "__REACTION_HUB_COMMAND_START__"
        marker_key = "__REACTION_HUB_EXIT_CODE__"
        wrapped_command = (
            f"printf '{start_marker}\\n'; printf '{start_marker}\\n' >&2; "
            f"{command}; __rh_ec=$?; printf '\\n{marker_key}=%s\\n' \"$__rh_ec\""
        )
        result = self.run(wrapped_command)

        stdout_text, rc_stdout = self._extract_marker(
            self._strip_before_marker(result.stdout, start_marker), marker_key
        )
        stderr_text, rc_stderr = self._extract_marker(
            self._strip_before_marker(result.stderr, start_marker), marker_key
        )
        marker_rc = rc_stdout if rc_stdout is not None else rc_stderr

        normalized_rc = self._normalize_returncode(result.returncode)
        effective_rc = marker_rc if marker_rc is not None else normalized_rc

        if effective_rc != 0:
            stderr_detail = self._strip_connection_closed_line(stderr_text).strip()
            stdout_detail = self._strip_connection_closed_line(stdout_text).strip()
            detail = stderr_detail or stdout_detail
            trimmed = self._trim_text(detail)
            hint = self._failure_hint(detail)
            raw_rc = result.returncode
            rc_line = f"{effective_rc}"
            if raw_rc != effective_rc:
                rc_line = f"{effective_rc} (raw={raw_rc})"
            message = (
                f"Remote command failed ({rc_line}).\n"
                f"SSH context: {self._mode_summary()}\n"
                f"Command: {command}\n"
                f"Remote output:\n{trimmed}"
            )
            if hint:
                message = f"{message}\nHint: {hint}"
            raise RuntimeError(message)
        return stdout_text

    def upload_text(self, remote_path: str, text: str):
        import base64
        path_q = shlex.quote(str(remote_path))
        try:
            if not text:
                self.exec_checked(f": > {path_q}")
                return
            b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
            res = self.run(f"base64 -d > {path_q}", input_text=b64, force_tty=False)
            if res.returncode != 0:
                raise RuntimeError(
                    f"Remote upload failed with exit code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Remote upload failed for path: {remote_path}\n"
                f"SSH context: {self._mode_summary()}\n"
                f"{exc}"
            ) from exc
