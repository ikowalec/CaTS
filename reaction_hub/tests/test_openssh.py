import os
from types import SimpleNamespace

from core.openssh import OpenSSHClient, SSHRunResult


def test_windows_ssh_uses_user_profile_known_hosts(monkeypatch):
    fake_os = SimpleNamespace(
        name="nt",
        environ={"USERPROFILE": r"C:\Users\researcher"},
        path=os.path,
    )
    monkeypatch.setattr("core.openssh.os", fake_os)

    args = OpenSSHClient(host="example.org", port=22, username="user")._base_args()

    assert r"UserKnownHostsFile=C:\Users\researcher\.ssh\known_hosts" in args


def test_exec_checked_discards_login_shell_output(monkeypatch):
    client = OpenSSHClient(host="example.org", port=22, username="user")

    def fake_run(command: str) -> SSHRunResult:
        assert "__REACTION_HUB_COMMAND_START__" in command
        return SSHRunResult(
            stdout=(
                "\x1b]P0000000\x1b]PFFFFFFF\r\n"
                "/home/user/.myenv: line 42: queue: command not found\r\n"
                "__REACTION_HUB_COMMAND_START__\r\n"
                "import json\r\n"
                "\r\n__REACTION_HUB_EXIT_CODE__=0\r\n"
            ),
            stderr="__REACTION_HUB_COMMAND_START__\n",
            returncode=0,
        )

    monkeypatch.setattr(client, "run", fake_run)

    assert client.exec_checked("cat run_workflow_submit.py") == "import json\n"


def test_exec_checked_keeps_stdout_error_when_stderr_only_closes_connection(monkeypatch):
    client = OpenSSHClient(host="example.org", port=22, username="user")

    monkeypatch.setattr(
        client,
        "run",
        lambda command: SSHRunResult(
            stdout=(
                "__REACTION_HUB_COMMAND_START__\n"
                "sbatch: error: invalid account\n"
                "__REACTION_HUB_EXIT_CODE__=1\n"
            ),
            stderr=(
                "__REACTION_HUB_COMMAND_START__\n"
                "Connection to example.org closed.\n"
            ),
            returncode=0,
        ),
    )

    try:
        client.exec_checked("sbatch submit_workflow.sh")
    except RuntimeError as exc:
        assert "sbatch: error: invalid account" in str(exc)
    else:
        raise AssertionError("Expected remote command failure")
