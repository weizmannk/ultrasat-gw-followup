{ echo '#!/bin/bash'
  echo "source $PWD/.venv/bin/activate"
  echo "cd $PWD"
  echo 'export TMPDIR=/local/$USER/cplex_tmp && mkdir -p $TMPDIR'
  condor_q -constraint 'JobStatus == 5' -af Cmd | sed 's/^/bash /'
} > rerun_held_pcdev6.sh
