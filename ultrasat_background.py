import json
import os
from pathlib import Path


import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy_healpix import HEALPix
from synphot import Observation, SourceSpectrum, exceptions as syn_ex
from synphot.models import ConstFlux1D

import galactic
import zodiacal
import exposure
from m4opt.missions import ultrasat

# Define paths for local resources
localpath = os.path.dirname(os.path.abspath("__file__"))
scaling_file_defaults = os.path.join(localpath, "scaling_default.txt")
with open(scaling_file_defaults, "r") as json_file:
    scaling = json.load(json_file)

# Paths to Zodiacal Light Spectrum files
ZODI_SPATIAL = os.path.join(localpath, "zodi_data", "Leinert97_table17.txt")
ZODI_SPEC = os.path.join(localpath, "zodi_data", "scaled_zodiacal_spec.txt")

# Telescope and observation parameters
NPIX = ultrasat.detector.npix
PIXEL = ultrasat.detector.plate_scale
AREA = ultrasat.detector.area

DARK_NOISE = ultrasat.detector.dark_noise
READ_NOISE = ultrasat.detector.read_noise
SNR = 10.0

nuv_band = ultrasat.detector.bandpasses["NUV"]

obstime = Time("2024-01-01 00:00:00")
nside = 64
ipix = 12290
m_obs = 28.310335426912324  # Observed magnitude in AB system

# Convert HEALPix index to sky coordinates
theta, phi = hp.pix2ang(nside, ipix, nest=True)
ra = np.rad2deg(phi)
dec = np.rad2deg(0.5 * np.pi - theta)
sky_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=ICRS, obstime=obstime)

# Galactic coordinates and spectrum
gal_coord = sky_coord.galactic
galactic_spec = galactic.galactic_nuv_spec(
    latitude=gal_coord.b,
    scaling=scaling,
    infile=ZODI_SPEC,
    pixel=PIXEL,
)
galactic_obs = Observation(galactic_spec, nuv_band, force="taper")

# Zodiacal spectrum and background  coord, time, zodi_spatial_file, zodi_spec_file, scaling
zodi_spec = zodiacal.zodi_spec_coords(
    coord=sky_coord,
    time=obstime,
    zodi_spatial_file=ZODI_SPATIAL,
    zodi_spec_file=ZODI_SPEC,
    scaling=scaling,
    pixel=PIXEL,
    diag=False,
)
zodi_obs = Observation(zodi_spec, nuv_band, force="taper")
sky_background = zodi_obs.countrate(area=AREA) + galactic_obs.countrate(area=AREA)

# Print the calculated sky background
print(f"Sky background countrate: {sky_back}")


# Calculate source count rate
source_spectrum = SourceSpectrum(ConstFlux1D, amplitude=m_obs * u.ABmag)
observation = Observation(source_spectrum, nuv_band)
source_rate = observation.countrate(area=AREA)

# Exposure time calculations
texp = exposure.get_exposure(
    source=source_rate,
    sky=sky_background,
    npix=NPIX,
    dark_noise=DARK_NOISE,
    read_noise=READ_NOISE,
    snr=10.0,
    neff=False,
)


print(f"Exposure time: {texp}")
