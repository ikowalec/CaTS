from __future__ import annotations

from dataclasses import dataclass
import shlex

from django.conf import settings

from core.openssh import OpenSSHClient


@dataclass
class SlurmJobSnapshot:
    job_id: str
    state: str
    runtime: str = ""
    name: str = ""
    workdir: str = ""
    exit_code: str = ""


class SlurmSSHClient:
    def __init__(self, host: str, port: int, username: str, ssh_key_path: str = ""):
        self.host = host
        self.port = port
        self.username = username
        self.ssh_key_path = ssh_key_path
        self._client: OpenSSHClient | None = None

    def __enter__(self) -> "SlurmSSHClient":
        client = OpenSSHClient(
            host=self.host,
            port=self.port,
            username=self.username,
            ssh_key_path=self.ssh_key_path,
            timeout=15,
            force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
            batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
            use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
            bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
        )
        self._client = client
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._client is not None:
            self._client.close()
            self._client = None

    def _run(self, command: str) -> tuple[str, str, int]:
        if self._client is None:
            raise RuntimeError("SSH client not connected")
        result = self._client.run(command)
        return result.stdout, result.stderr, result.returncode

    def exec_checked(self, command: str) -> str:
        if self._client is None:
            raise RuntimeError("SSH client not connected")
        return self._client.exec_checked(command)

    def file_exists(self, remote_path: str) -> bool:
        path_q = shlex.quote(str(remote_path))
        out = self.exec_checked(f"if [ -f {path_q} ]; then echo 1; else echo 0; fi")
        return out.strip() == "1"

    def directory_exists(self, remote_path: str) -> bool:
        path_q = shlex.quote(str(remote_path))
        out = self.exec_checked(f"if [ -d {path_q} ]; then echo 1; else echo 0; fi")
        return out.strip() == "1"

    def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
        unique_paths: list[str] = []
        seen: set[str] = set()
        for raw in remote_paths:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            unique_paths.append(value)
        if not unique_paths:
            return {}

        results: dict[str, bool] = {}
        chunk_size = 128
        for start in range(0, len(unique_paths), chunk_size):
            chunk = unique_paths[start : start + chunk_size]
            quoted = " ".join(shlex.quote(path) for path in chunk)
            cmd = (
                f"for p in {quoted}; do "
                "if [ -d \"$p\" ]; then printf '%s|1\\n' \"$p\"; "
                "else printf '%s|0\\n' \"$p\"; fi; "
                "done"
            )
            out = self.exec_checked(cmd)
            for line in out.splitlines():
                text = line.strip()
                if not text or "|" not in text:
                    continue
                path_text, flag = text.rsplit("|", 1)
                results[path_text] = flag.strip() == "1"

        for path in unique_paths:
            if path in results:
                continue
            try:
                results[path] = self.directory_exists(path)
            except Exception:
                continue
        return results

    def read_text(self, remote_path: str) -> str:
        path_q = shlex.quote(str(remote_path))
        return self.exec_checked(f"cat {path_q}")

    def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
        if not job_ids:
            return {}

        def _is_invalid_job_lookup_error(text: str) -> bool:
            lowered = str(text or "").strip().lower()
            if not lowered:
                return False
            return "invalid job id" in lowered or "slurm_load_jobs error" in lowered

        def _sacct_job_id_candidates(raw_job_id: str) -> list[str]:
            job_id_text = str(raw_job_id or "").strip()
            if not job_id_text:
                return []

            candidates: list[str] = []
            base = job_id_text.split(".", 1)[0]
            plus_trimmed = base.split("+", 1)[0]
            candidates.extend([job_id_text, base, plus_trimmed])
            if "_" in plus_trimmed:
                array_root = plus_trimmed.split("_", 1)[0]
                candidates.append(array_root)

            unique: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                value = str(candidate or "").strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                unique.append(value)
            return unique

        def _parse_scontrol_snapshot(line: str) -> dict[str, str]:
            fields: dict[str, str] = {}
            for token in str(line or "").strip().split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                fields[key] = value
            job_id_text = str(fields.get("JobId") or fields.get("JobIdRaw") or "").strip()
            if not job_id_text:
                return {}
            return {
                "job_id": job_id_text,
                "state": str(fields.get("JobState") or "").strip(),
                "runtime": str(fields.get("RunTime") or "").strip(),
                "name": str(fields.get("JobName") or "").strip(),
                "workdir": str(fields.get("WorkDir") or "").strip(),
                "exit_code": str(fields.get("ExitCode") or "").strip(),
            }

        username_flag = f"-u {self._username}" if getattr(self, "_username", "") else ""
        cmd = f"squeue {username_flag} -h -o '%i|%T|%M|%j|%Z'"
        out, err, code = self._run(cmd)
        if code != 0:
            id_csv = ",".join(sorted(set(job_ids)))
            cmd = f"squeue -h -o '%i|%T|%M|%j|%Z' -j {id_csv}"
            out, err, code = self._run(cmd)
            if code != 0:
                combined_error = "\n".join(part for part in (str(err or "").strip(), str(out or "").strip()) if part)
                if not _is_invalid_job_lookup_error(combined_error):
                    raise RuntimeError(f"squeue failed: {combined_error}")

        snapshots: dict[str, dict] = {}
        for line in out.splitlines():
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            job_id, state, runtime, name, workdir = parts[:5]
            snapshots[job_id] = {
                "job_id": job_id,
                "state": state,
                "runtime": runtime,
                "name": name,
                "workdir": workdir,
            }

        missing = [jid for jid in job_ids if jid not in snapshots]
        if missing:
            sacct_cmd = f"sacct {username_flag} -n -P -X -o JobID,State,Elapsed,JobName,WorkDir,ExitCode"
            sacct_out, _sacct_err, sacct_code = self._run(sacct_cmd)
            if sacct_code != 0:
                missing_csv = ",".join(sorted(set(missing)))
                sacct_cmd = f"sacct -n -P -j {missing_csv} -o JobID,State,Elapsed,JobName,WorkDir,ExitCode"
                sacct_out, _sacct_err, sacct_code = self._run(sacct_cmd)
                if sacct_code != 0:
                    sacct_out = ""
            missing_set = set(missing)
            selected_rows: dict[str, tuple[tuple[int, int], dict[str, str]]] = {}

            def _state_rank(raw_state: str) -> int:
                state = (raw_state or "").strip().upper()
                if not state:
                    return 0
                primary = state.split()[0]
                if primary in {"F", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
                    return 40
                if primary in {"CA", "CANCELLED", "CANCELLED+"} or primary.startswith("CANCELLED"):
                    return 35
                if primary in {"R", "RUNNING", "COMPLETING"}:
                    return 30
                if primary in {"PD", "PENDING", "CONFIGURING", "S", "SUSPENDED"}:
                    return 20
                if primary in {"CD", "COMPLETED"}:
                    return 10
                return 0

            for line in sacct_out.splitlines():
                parts = line.strip().split("|")
                if len(parts) < 6:
                    continue
                job_id, state, runtime, name, workdir, exit_code = parts[:6]
                job_id_text = str(job_id or "").strip()
                if not job_id_text:
                    continue
                matching_job_ids = [
                    candidate for candidate in _sacct_job_id_candidates(job_id_text) if candidate in missing_set
                ]
                if not matching_job_ids:
                    continue
                base_job_id = matching_job_ids[0]

                # Prefer the exact parent row (e.g. "12585898") over sub-rows
                # (e.g. "12585898.batch", "12585898.0"), and within each class
                # prefer more severe states.
                score = (1 if job_id_text == base_job_id else 0, _state_rank(state))
                payload = {
                    "job_id": base_job_id,
                    "state": state,
                    "runtime": runtime,
                    "name": name,
                    "workdir": workdir,
                    "exit_code": exit_code,
                }
                current = selected_rows.get(base_job_id)
                if current is None or score > current[0]:
                    selected_rows[base_job_id] = (score, payload)

            for base_job_id, (_score, payload) in selected_rows.items():
                snapshots[base_job_id] = payload

            unresolved_job_ids = [job_id for job_id in missing if job_id not in snapshots]
            for unresolved_job_id in unresolved_job_ids:
                scontrol_cmd = f"scontrol -o show job {shlex.quote(unresolved_job_id)}"
                scontrol_out, _scontrol_err, scontrol_code = self._run(scontrol_cmd)
                if scontrol_code != 0:
                    continue
                for line in scontrol_out.splitlines():
                    snapshot = _parse_scontrol_snapshot(line)
                    if not snapshot:
                        continue
                    matching_job_ids = [
                        candidate
                        for candidate in _sacct_job_id_candidates(snapshot.get("job_id", ""))
                        if candidate in missing_set
                    ]
                    if not matching_job_ids:
                        continue
                    base_job_id = matching_job_ids[0]
                    snapshot["job_id"] = base_job_id
                    snapshots[base_job_id] = snapshot
                    break

        return snapshots
