#!/bin/bash

docker run -it \
    -v /home/ec2-user/data:/usr/src/app/data \
    -v /home/ec2-user/models:/usr/src/app/models \
    -v /home/ec2-user/logs:/usr/src/app/logs \
    jimfawkes/project-fletcher:latest \
    python3 fletcher --help

