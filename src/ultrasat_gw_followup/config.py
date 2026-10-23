"""Read a run's ``.ini`` file and turn its values into argparse defaults.

Every workflow step takes ``--config config/<run>.ini``. The file supplies the
defaults; an option given on the command line replaces them. A key that is
missing, blank or the literal ``none`` leaves the step's own default in place,
which for the m4opt pass-through options means the flag is omitted and m4opt
applies its default.

Usage
-----
::

    parser = argparse.ArgumentParser()
    config, argv = Config.from_command_line(parser)
    parser.add_argument("--snr", type=int, default=config.getint("scheduling", "snr", 5))
    args = parser.parse_args(argv)
"""

import configparser
from pathlib import Path

#: Spellings of "this key says nothing; keep the default".
EMPTY = frozenset({"", "none"})

#: Spellings of a boolean, as accepted in the ``.ini`` files.
TRUE = frozenset({"1", "true", "yes", "on"})
FALSE = frozenset({"0", "false", "no", "off"})


class Config:
    """The values of one ``.ini`` file, or an empty set of values.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        File to read. ``None`` builds an empty configuration, so that a step
        run without ``--config`` goes through the same code path.

    Raises
    ------
    FileNotFoundError
        If ``path`` is given but cannot be read. ``configparser`` ignores an
        unreadable path silently, which would turn every setting into its
        default and only surface much later as a wrong m4opt argument.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
        if path is not None and not self._parser.read(path):
            raise FileNotFoundError(f"config file not found or unreadable: {path}")

    @classmethod
    def from_command_line(cls, parser, argv=None):
        """Consume ``--config`` from the command line and read that file.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            Parser of the calling step. ``--config`` is added to it, so that
            it appears in ``--help``.
        argv : sequence of str, optional
            Arguments to parse; defaults to ``sys.argv[1:]``.

        Returns
        -------
        config : Config
            Values of the file, or an empty configuration.
        remaining : list of str
            Arguments left for the caller's own parser.
        """
        parser.add_argument("--config", type=Path, help="path to a .ini config file")
        known, remaining = parser.parse_known_args(argv)
        try:
            return cls(known.config), remaining
        except FileNotFoundError as error:
            parser.error(str(error))

    def get(self, section, key, fallback=None):
        """Return a value as a string, or ``fallback`` if it says nothing."""
        value = self._parser.get(section, key, fallback=None)
        if value is None or value.strip().lower() in EMPTY:
            return fallback
        return value.strip()

    def getint(self, section, key, fallback=None):
        """Return a value as an int, or ``fallback`` if it says nothing."""
        value = self.get(section, key)
        return fallback if value is None else int(value)

    def getfloat(self, section, key, fallback=None):
        """Return a value as a float, or ``fallback`` if it says nothing.

        ``inf`` is accepted, and is how ``exptime_max`` is left unconstrained.
        """
        value = self.get(section, key)
        return fallback if value is None else float(value)

    def getbool(self, section, key, fallback=None):
        """Return a value as a bool, or ``fallback`` if it says nothing.

        Raises
        ------
        ValueError
            If the value is neither a true nor a false spelling.
        """
        value = self.get(section, key)
        if value is None:
            return fallback
        lowered = value.lower()
        if lowered in TRUE:
            return True
        if lowered in FALSE:
            return False
        raise ValueError(f"[{section}] {key}: not a boolean: {value!r}")

    def getlist(self, section, key, fallback=None):
        """Return a whitespace-separated value as a list of strings."""
        value = self.get(section, key)
        return fallback if value is None else value.split()

    def getsection(self, section):
        """Return every non-empty key of a section, as a dict of strings.

        Sections whose keys are passed straight through to a batch system
        ([slurm], [condor]) are read this way, so that nothing has to be
        hard-coded in the backend.
        """
        if not self._parser.has_section(section):
            return {}
        values = ((key, self.get(section, key)) for key in self._parser[section])
        return {key: value for key, value in values if value is not None}


def resolve_input(path, base):
    """Resolve an input file against a base directory.

    A path that already points at an existing file is taken as given; anything
    else is read from ``base``. This is what lets ``--event-table`` name either
    the shared ``observing-scenarios.ecsv`` inside ``data_dir`` or a
    ``redo.ecsv`` written elsewhere by :mod:`~ultrasat_gw_followup.workflow.make_redo`.

    Parameters
    ----------
    path : str or pathlib.Path
        File name or path.
    base : str or pathlib.Path
        Directory to fall back on.

    Returns
    -------
    pathlib.Path
        The resolved path.
    """
    candidate = Path(path)
    return candidate if candidate.is_file() else Path(base) / candidate
