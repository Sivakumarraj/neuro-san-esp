"""Exercise the installed command's argument and environment handling."""

import os

import pytest

from esp.cli import REPORTS, ROOT, Cli


def test_parser_lists_every_command():
    parser = Cli.parser()
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["--help"])
    help_text = parser.format_help()
    for command in ("validate", "lint", "hocon", "test", "offline-check", "offline", "check-key",
                    "smoke", "measure", "loop", "studio", "web", "champion", "report"):
        assert command in help_text


def test_environment_prepends_root_using_platform_separator(monkeypatch):
    monkeypatch.setattr("esp.config.bootstrap", lambda: None)
    previous = os.pathsep.join(("existing", "another"))
    monkeypatch.setenv("PYTHONPATH", previous)
    monkeypatch.delenv("AGENT_TOOL_PATH", raising=False)
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    env = Cli().environment()
    assert env["PYTHONPATH"] == os.pathsep.join((str(ROOT), previous))
    assert env["AGENT_TOOL_PATH"] == str(ROOT)
    assert env["PYTHONUTF8"] == "1"
    monkeypatch.delenv("PYTHONPATH")
    assert Cli().environment()["PYTHONPATH"] == str(ROOT)


def test_report_paths_exist():
    assert all((ROOT / path).is_file() for path in REPORTS.values())


@pytest.mark.parametrize(("argv", "expected"), [
    (["test", "--cov", "-k", "cross platform"],
     ("-m", "pytest", "tests", "-q", "--cov", "-k", "cross platform")),
    (["loop", "--live", "--budget", "5"], ("-m", "esp.loop", "--live", "--budget", "5")),
])
def test_flags_reach_the_downstream_parser(monkeypatch, argv, expected):
    cli = Cli()

    def run(*args):
        assert args == expected
        return 7

    monkeypatch.setattr(cli, "run", run)
    assert cli.main(argv) == 7
