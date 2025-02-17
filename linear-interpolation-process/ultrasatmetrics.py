# ULTRASAT SNR Module
"""
ULTRASAT SNR Module
---------------
This module provides tools to calculate signal-to-noise ratio (SNR),
required exposure time, and limiting magnitudes for ULTRASAT observations.
"""

from astropy.stats import signal_to_noise_oir_ccd
import numpy as np
import astropy.units as u
from m4opt.missions import ultrasat

# Detector constants
NPIX = ultrasat.detector.npix
DARK_NOISE = ultrasat.detector.dark_noise
READ_NOISE = ultrasat.detector.read_noise
PIXEL_SCALE = ultrasat.detector.plate_scale
AREA = ultrasat.detector.area


def calc_exposure(k, src_rate, bgd_rate, read_noise, neff):
    """
    Compute the time to reach a given SNR (k) given the source rate, background rate, read noise,
    and number of effective background pixels.

    Parameters
    ----------
    k : float
        Desired signal-to-noise ratio.
    src_rate : float
        Source count rate in counts per second.
    bgd_rate : float
        Background count rate in counts per second.
    read_noise : float
        Read noise per pixel.
    neff : float
        Number of effective pixels.

    Returns
    -------
    float
        Required exposure time in seconds.
    """
    denom = 2 * src_rate**2
    nom1 = (k**2) * (src_rate + neff * bgd_rate)
    nom2 = (
        k**4 * (src_rate + neff * bgd_rate) ** 2
        + 4 * k**2 * src_rate**2 * neff * read_noise**2
    ) ** 0.5
    exposure = (nom1 + nom2) / denom
    return exposure


def get_exposure(
    source,
    sky,
    npix=NPIX,
    dark_noise=DARK_NOISE,
    read_noise=READ_NOISE,
    snr=10.0,
    neff=False,
):
    """
    Calculate the required exposure time to achieve a specified SNR.

    Parameters
    ----------
    source : astropy.units.Quantity
        Source count rate.
    sky : astropy.units.Quantity
        Sky count rate.
    npix : float, optional
        Number of effective pixels. Defaults to NPIX.
    dark_noise : astropy.units.Quantity, optional
        Dark noise in counts per second. Defaults to DARK_NOISE.
    read_noise : float, optional
        Read noise per pixel. Defaults to READ_NOISE.
    snr : float, optional
        Desired signal-to-noise ratio. Defaults to 10.0.
    neff : float, optional
        Number of effective pixels. If not provided, defaults to npix.

    Returns
    -------
    float
        Required exposure time in seconds.
    """
    if neff is False:
        neff = npix

    return calc_exposure(
        snr, source.value, sky.value + dark_noise.value, read_noise, neff
    )


def get_version():
    """
    Retrieve the version of the ULTRASAT package.

    Returns
    -------
    str
        The version string.
    """
    from pkg_resources import get_distribution

    return get_distribution("ultrasat").version


# Example usage
# if __name__ == "__main__":
#     # Example data for testing
#     source_rate = 10 * u.ct / u.s
#     sky_rate = 5 * u.ct / u.s
#     snr_target = 5.0

#     exposure_time = get_exposure(source_rate, sky_rate, snr=snr_target)
#     print(f"Required exposure time for SNR={snr_target}: {exposure_time:.2f} seconds")
