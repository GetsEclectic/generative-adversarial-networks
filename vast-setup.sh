#!/usr/bin/env bash

# first argument is vast ssh host number
# second argument is vast ssh port
# eg
# ./vast-setup 5 10444
# for ssh5.vast.ai:10444

ssh -p $VAST_PORT root@ssh$VAST_HOST_NUM.vast.ai -L 8080:localhost:8080 "DEBIAN_FRONTEND=noninteractive apt-get -yq install rsync"
rsync -a -e "ssh -p $VAST_PORT" host-setup.sh stylegan2-pytorch root@ssh$VAST_HOST_NUM.vast.ai:
ssh -p $VAST_PORT root@ssh$VAST_HOST_NUM.vast.ai -L 8080:localhost:8080 "chmod 774 host-setup.sh; ./host-setup.sh"
ssh -p $VAST_PORT root@ssh$VAST_HOST_NUM.vast.ai -L 8080:localhost:8080
