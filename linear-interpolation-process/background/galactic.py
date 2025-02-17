import astropy.units as u
import numpy as np
import os
from specutils import Spectrum1D

# Characteristics

# Pixel size:
PIX_UM = 10 * u.micron

# Plate scale for imager and spectrometer
PLATE_SCALE = 1.03 * u.arcsec / PIX_UM
SPEC_PLATE_SCALE = 0.8 * u.arcsec / PIX_UM

# Pixel area
PIXEL = (PIX_UM * PLATE_SCALE) ** 2


def diffuse_galactic_nuv_norm(latitude):
    """
    Generates NUV flux from Table 4 of Murthy et al (2014).

    NOTE: Not applicable for observations in the plane

    Parameters
    -----------
    latitude : Astropy quantity or array of Astropy quantities

    Returns
    --------
    fuv_flux : Astropy quantity with PHLAM units

    """
    from numpy import sin, abs

    south = latitude < -0 * u.deg
    north = latitude >= 0 * u.deg

    fuv_flux = np.zeros_like(latitude.value)

    fuv_flux[north] = 257.5 + 185.1 / sin(abs(latitude[north]))
    fuv_flux[south] = 66.7 + 356.3 / sin(abs(latitude[south]))

    fuv_flux *= u.ph / (u.cm**2 * u.sr * u.s * u.AA)

    return fuv_flux


def galactic_nuv_spec(
    latitude, scaling, infile="scaled_zodiacal_spec.txt", pixel=PIXEL
):
    """
    Takes as input the galactic latitude and generates a flat (in photon units)
    spectrum with flux scaled from Murthy (2014)

    Parameters
    -----------
    latitude : Astropy quantity (degrees)
        Latitude of the observation

    Optional Parameters
    -------------------
    infile : str
        File to use to define the input wavelength range. Defaults to Zodi.

    Returns
    -------
    Spectrum1D object
    """

    # Check latitude validity (outside of 15 degrees from the plane)
    # if np.any(abs(latitude) < 15 * u.deg):
    #     raise ValueError("Latitude must be more than 15 degrees away from the galactic plane.")

    # Use the Zodi file
    data = np.genfromtxt(infile)

    # Get the scale term:
    scale = diffuse_galactic_nuv_norm(latitude)
    scale *= 1.0 + scaling["galactic_scattered_light"]

    wave = data[:, 0] * u.AA
    flux = np.zeros_like(wave.value)
    flux += scale.value
    flux *= scale.unit

    # Convert to per-pixel units
    flux = flux.to(u.ph / (u.cm**2 * u.Angstrom * u.arcsec**2 * u.s))
    flux *= pixel

    return Spectrum1D(flux=flux, spectral_axis=wave)
