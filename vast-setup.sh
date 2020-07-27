#!/usr/bin/env bash

# first argument is vast ssh host number
# second argument is vast ssh port
# eg
# ./vast-setup 5 10444
# for ssh5.vast.ai:10444

ssh -p $2 root@ssh$1.vast.ai -L 8080:localhost:8080 "DEBIAN_FRONTEND=noninteractive apt-get -yq install rsync"
rsync -a -e "ssh -p $2" host-setup.sh stylegan2-pytorch root@ssh$1.vast.ai:/workspace
ssh -p $2 root@ssh$1.vast.ai -L 8080:localhost:8080 "cd /workspace; chmod 774 host-setup.sh; ./host-setup.sh"
ssh -p $2 root@ssh$1.vast.ai -L 8080:localhost:8080
