#!/bin/bash
# watch -n 1 nvidia-smi
echo $HOSTNAME
wait 20
python -u q3_1_gan.py