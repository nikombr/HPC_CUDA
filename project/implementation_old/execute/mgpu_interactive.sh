bsub -q gpuh100i -app h100app -gpu "num=2:mode=shared" -n 8 -R "rusage[mem=2G]" -R "span[hosts=1]"  -Is -J qrsh "/bin/bash -l"


## bsub -q gpuh100i -app h100app -gpu "num=2:mode=shared"  -env TERM,LSF_QRSH -Is -J qrsh "reset; /bin/bash -l"