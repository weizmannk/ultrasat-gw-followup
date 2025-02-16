import numpy as np


from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.skymap import distance
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from scipy import stats


# from m4opt.missions import ultrasat, uvex
# from m4opt.models import observing, DustExtinction

skymap = (
    "/home/weizmann.kiendrebeogo/M4OPT-ULTRSASAT/runs_SNR-10/O5HLVK/farah/allsky/0.fits"
)

time_step = 1 * u.minute
delay = 6 * u.hour
deadline = 24 * u.hour
nside = 128


hpx = HEALPix(nside, frame=ICRS(), order="nested")

skymap_moc = read_sky_map(skymap, moc=True)
skymap_flat = rasterize(skymap_moc, hpx.level)

event_time = Time(Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso")

distmean, diststd, _ = distance.parameters_to_moments(
    skymap_flat["DISTMU"], skymap_flat["DISTSIGMA"]
)


absmag_mean = -16
absmag_stdev = 0

logdist_sigma2 = np.log1p(np.square(diststd / distmean))
logdist_sigma = np.sqrt(logdist_sigma2)
logdist_mu = np.log(distmean) - 0.5 * logdist_sigma2
a = 5 / np.log(10)

appmag_mu = absmag_mean + a * logdist_mu + 25
appmag_sigma = np.sqrt(np.square(absmag_stdev) + np.square(a * logdist_sigma))
quantiles = np.linspace(0.05, 0.95, 5)
appmag_quantiles = stats.norm(
    loc=appmag_mu[:, np.newaxis], scale=appmag_sigma[:, np.newaxis]
).ppf(quantiles)
appmag_quantiles[0]
