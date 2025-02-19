#!/bin/bash

/home/weizmann.kiendrebeogo/anaconda3/envs/new-m4opt/bin/m4opt schedule /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/runs_SNR-10/O5HLVK/farah/allsky/6220.fits /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/schedules/6220.ecsv --mission=ultrasat --bandpass=NUV --absmag-mean=-16.0 --absmag-stdev=1.0 --exptime-min='300 s' --exptime-max='14400.0 s' --snr=10 --delay='15 min' --deadline='24 hr' --timelimit='20min' --nside=128 --write-progress /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/progress/PROGRESS_6220.ecsv --jobs 0





m4opt animate /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/schedules/6220.ecsv  6220_MOVIE.gif --dpi 300 --still  6220.pdf

m4opt animate /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/schedules/339.ecsv  339_MOVIE.gif --dpi 300 --still  339.pdf




m4opt animate /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/schedules/5578.ecsv  5578_MOVIE.gif --dpi 300 --still   5578.pdf


m4opt animate /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/BNS_NSBH/O5HLVK/schedules/106.ecsv 106_MOVIE.gif --dpi 300 --still   106.pdf


 ligo-skymap-plot  ./data/5578.fits  -o ./data/5578_skymap.png  --annotate --contour 90 --colorbar --inj-database /home/weizmann.kiendrebeogo/M4OPT-ULTRASAT/ultrasat-gw-followup/runs_SNR-10/O5HLVK/farah/events.sqlite



 import missions

  missions.Mission = getattr(missions, table.meta["args"]["mission"])
