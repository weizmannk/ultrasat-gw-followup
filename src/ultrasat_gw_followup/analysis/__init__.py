"""Tables and figures derived from the post-processed event tables."""


def get_skygrid(table):
    """Return the sky grid name of an event table, or ``""`` if it has none.

    Missions with several survey grids (ULTRASAT: ``allsky``,
    ``non-overlap``) write the grid name into every row. Missions with a
    single pointing strategy (UVEX) have no grid to name, and postprocess
    writes a column of Python ``None`` there; some tables instead hold the
    literal string ``"None"``. Both spell "no grid" and are normalised to
    ``""``.

    Parameters
    ----------
    table : astropy.table.Table
        Event table carrying a ``skygrid`` column.

    Returns
    -------
    str
        The grid name, or ``""`` when the mission has no grid variants.

    Raises
    ------
    ValueError
        If the table mixes several grids, which would make a single caption
        or file name wrong for part of the rows.
    """
    values = {
        "" if value is None or value == "None" else str(value)
        for value in table["skygrid"]
    }
    if len(values) > 1:
        raise ValueError(f"several skygrid values in one table: {sorted(values)}")
    return values.pop() if values else ""
