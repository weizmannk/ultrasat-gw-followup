import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy_healpix import HEALPix
from synphot import Observation, SourceSpectrum, exceptions as syn_ex
from synphot.models import ConstFlux1D

from background import galactic, zodiacal
import ultrasatmetrics
from m4opt.missions import uvex

import json
import os
from pathlib import Path

# Define paths for local resources
localpath = os.path.dirname(os.path.abspath("__file__"))
scaling_file_defaults = os.path.join(localpath, "scaling_default.txt")
with open(scaling_file_defaults, "r") as json_file:
    scaling = json.load(json_file)

# Paths to Zodiacal Light Spectrum files
ZODI_SPATIAL = os.path.join(localpath, "background/data", "Leinert97_table17.txt")
ZODI_SPEC = os.path.join(localpath, "background/data", "scaled_zodiacal_spec.txt")

# Telescope and observation parameters
NPIX = 10.5  # uvex.detector.npix  # 10.15
# PIXEL = uvex.detector.plate_scale
AREA = uvex.detector.area

# Pixel size:
PIX_UM = 10 * u.micron

# Plate scale for imager and spectrometer
PLATE_SCALE = 1.03 * u.arcsec / PIX_UM
SPEC_PLATE_SCALE = 0.8 * u.arcsec / PIX_UM

# Pixel area
PIXEL = (PIX_UM * PLATE_SCALE) ** 2


DARK_NOISE = uvex.detector.dark_noise
READ_NOISE = uvex.detector.read_noise

bandpass = uvex.detector.bandpasses["NUV"]

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
    latitude=gal_coord.b, scaling=scaling, infile=ZODI_SPEC, pixel=PIXEL
)
galactic_obs = Observation(galactic_spec, bandpass)

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
zodi_obs = Observation(zodi_spec, bandpass)
sky_background = zodi_obs.countrate(area=AREA) + galactic_obs.countrate(area=AREA)

# Print the calculated sky background
print(f"Sky background countrate: {sky_background}")


# Calculate source count rate
source_spectrum = SourceSpectrum(ConstFlux1D, amplitude=m_obs * u.ABmag)
observation = Observation(source_spectrum, bandpass)
source_rate = observation.countrate(area=AREA)

# Exposure time calculations
texp = ultrasatmetrics.get_exposure(
    source=source_rate,
    sky=sky_background,
    npix=NPIX,
    dark_noise=DARK_NOISE,
    read_noise=READ_NOISE,
    snr=5.0,
    neff=False,
)


print(f"Exposure time: {texp}")

# Observation(galactic_spec, bandpass, force='extrap')


# ############################
# # With M4OPT
###########################

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import ICRS, SkyCoord, EarthLocation
from astropy_healpix import HEALPix

from synphot import Observation
from synphot import ConstFlux1D, SourceSpectrum
from m4opt.missions import uvex
from m4opt.models import observing

# Observation parameters
obstime = Time("2024-01-01 00:00:00")
nside = 64  # Healpix resolution
ipix = 12290  # Healpix pixel index
m_obs = 28.310335426912324  # AB magnitude
SNR = 5.0

theta, phi = hp.pix2ang(nside, ipix, nest=True)
ra = np.rad2deg(phi)
dec = np.rad2deg(0.5 * np.pi - theta)
sky_coord = SkyCoord(ICRS(ra=ra * u.deg, dec=dec * u.deg), obstime=obstime)

observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)

with observing(
    observer_location=observer_location,
    target_coord=sky_coord,
    obstime=obstime,
):
    texp = uvex.detector.get_exptime(
        snr=SNR,
        source_spectrum=SourceSpectrum(ConstFlux1D, amplitude=m_obs * u.ABmag),
        bandpass="NUV",
    )

print("exposure time: ", texp)
