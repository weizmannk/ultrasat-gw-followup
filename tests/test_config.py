"""The .ini reader: typed access, blank values, and command-line precedence.

The contract the workflow steps rely on is that a key which says nothing --
missing, blank, or the literal ``none`` -- leaves the caller's default in
place, and that an option given on the command line wins over the file.
"""

import argparse
from pathlib import Path

import pytest

from ultrasat_gw_followup.config import Config, resolve_input

INI = """\
[scheduling]
snr         = 5
exptime_max = inf
visits      = 2
appmag_dist = true
max_fields  =
cadence     = none
[mission]
skygrid = non-overlap
[scenarios]
runs = O5a O5b O5c
[slurm]
partition = cpu
account   =
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "run.ini"
    path.write_text(INI)
    return Config(path)


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config(tmp_path / "absent.ini")


def test_no_path_gives_an_empty_configuration():
    assert Config().get("scheduling", "snr", "fallback") == "fallback"


@pytest.mark.parametrize("key", ["max_fields", "cadence", "absent"])
def test_a_key_that_says_nothing_keeps_the_default(config, key):
    assert config.get("scheduling", key) is None
    assert config.get("scheduling", key, "kept") == "kept"
    assert config.getint("scheduling", key, 7) == 7


def test_typed_access(config):
    assert config.getint("scheduling", "snr") == 5
    assert config.getfloat("scheduling", "exptime_max") == float("inf")
    assert config.getbool("scheduling", "appmag_dist") is True
    assert config.getlist("scenarios", "runs") == ["O5a", "O5b", "O5c"]


def test_a_non_boolean_is_rejected(tmp_path):
    path = tmp_path / "bad.ini"
    path.write_text("[scheduling]\nappmag_dist = sometimes\n")
    with pytest.raises(ValueError, match="not a boolean"):
        Config(path).getbool("scheduling", "appmag_dist")


def test_getsection_drops_the_keys_that_say_nothing(config):
    assert config.getsection("slurm") == {"partition": "cpu"}
    assert config.getsection("absent") == {}


def test_the_command_line_wins_over_the_file(tmp_path):
    path = tmp_path / "run.ini"
    path.write_text(INI)

    def parse(argv):
        parser = argparse.ArgumentParser()
        config, remaining = Config.from_command_line(parser, argv)
        parser.add_argument(
            "--snr", type=int, default=config.getint("scheduling", "snr")
        )
        return parser.parse_args(remaining)

    assert parse(["--config", str(path)]).snr == 5
    assert parse(["--config", str(path), "--snr", "8"]).snr == 8


def test_from_command_line_reports_an_unreadable_file(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        Config.from_command_line(parser, ["--config", str(tmp_path / "absent.ini")])
    assert "not found or unreadable" in capsys.readouterr().err


def test_resolve_input_prefers_an_existing_path(tmp_path):
    here = tmp_path / "redo.ecsv"
    here.write_text("")
    assert resolve_input(here, tmp_path / "data") == here
    assert resolve_input("events.ecsv", tmp_path) == Path(tmp_path) / "events.ecsv"
