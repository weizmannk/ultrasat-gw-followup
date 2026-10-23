"""Import and entry-point smoke test."""

from importlib.metadata import entry_points

import ultrasat_gw_followup

STEPS = (
    "ugf-download",
    "ugf-unpacker",
    "ugf-scheduler",
    "ugf-postprocess",
    "ugf-check-runs",
    "ugf-make-redo",
    "ugf-paper-results",
)


def test_version_is_exposed():
    assert isinstance(ultrasat_gw_followup.__version__, str)
    assert ultrasat_gw_followup.__version__[0].isdigit()


def test_every_step_is_installed_as_a_command():
    installed = {entry.name: entry for entry in entry_points(group="console_scripts")}
    for name in STEPS:
        assert name in installed, name
        assert callable(installed[name].load())
