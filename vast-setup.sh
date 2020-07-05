#!/usr/bin/env bash

# first argument is vast ssh host number
# second argument is vast ssh port
# eg
# ./vast-setup 5 10444
# for ssh5.vast.ai:10444

scp -P $2 host-setup.sh root@ssh$1.vast.ai:
ssh -p $2 root@ssh$1.vast.ai -L 8080:localhost:8080
