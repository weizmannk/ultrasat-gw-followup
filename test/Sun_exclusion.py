import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import get_body, SkyCoord, EarthLocation
from astropy import units as u

# ---- Step 1: Define observation parameters ----
obstime = Time("2025-02-15T12:00:00")  # Observation time
location = EarthLocation.of_site("Las Campanas Observatory")  # Telescope location
min_sun_separation = 70 * u.deg  # Sun exclusion radius (adjustable)

# ---- Step 2: Compute Sun's position ----
sun_coord = get_body("sun", obstime, location)

# ---- Step 3: Generate random observation targets (RA, Dec) ----
num_targets = 500
target_ra = np.random.uniform(0, 360, num_targets)  # RA in degrees
target_dec = np.random.uniform(-90, 90, num_targets)  # Dec in degrees
targets = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

# ---- Step 4: Compute separation from the Sun ----
separation = targets.separation(sun_coord)

# Identify targets inside and outside Sun exclusion zone
inside_sun_exclusion = separation < min_sun_separation
outside_sun_exclusion = ~inside_sun_exclusion

# ---- Step 5: Convert to HEALPix coordinates ----
nside = 64  # Resolution parameter (higher = finer map)
hp_map = np.zeros(hp.nside2npix(nside))  # Empty HEALPix map

# Convert (RA, Dec) to HEALPix theta, phi
theta = np.radians(90 - target_dec)  # Convert Dec to colatitude
phi = np.radians(target_ra)  # Convert RA to longitude

# Convert Sun coordinates to HEALPix format
sun_theta = np.radians(90 - sun_coord.dec.deg)
sun_phi = np.radians(sun_coord.ra.deg)

# ---- Step 6: Assign values to HEALPix map ----
for i in range(num_targets):
    pix_idx = hp.ang2pix(nside, theta[i], phi[i])
    hp_map[pix_idx] = 1 if inside_sun_exclusion[i] else 0.5  # Mark Sun exclusion region

# ---- Step 7: Plot the HEALPix map ----
plt.figure(figsize=(10, 6))
hp.mollview(
    hp_map,
    title="GW-style Sky Map with Sun Exclusion Zone",
    unit="Visibility",
    cmap="coolwarm",
)
hp.graticule(color="white")

# Mark the Sun position
hp.projplot(sun_phi, sun_theta, "yo", markersize=10, label="Sun")

# Show the plot
plt.legend()
plt.show()
