#!/bin/bash

#Clear project source code

sudo rm -rf /home/ubuntu/projects/recommender_system_api

cd /home/ubuntu/git/recommender_system_api
eval `ssh-agent`
sudo chmod 400 /home/ubuntu/.ssh/id_rsa
ssh-add /home/ubuntu/.ssh/id_rsa

remote_repo=git@bitbucket.org:teecoin/recommender_system_api.git
local_repo=/home/ubuntu/git/recommender_system_api/

if [ -d $local_repo/.git ]; then pushd $local_repo; git pull; popd; else git clone $remote_repo . -b dev --depth 1; fi

#Copy source code to project folder
sudo rsync -avz --exclude '.git' /home/ubuntu/git/recommender_system_api/ /home/ubuntu/projects/recommender_system_api

#Run migrate
/home/ubuntu/.virtualenv/recommender_system_api/bin/pip install -r /home/ubuntu/projects/recommender_system_api/requirements.txt
/home/ubuntu/.virtualenv/recommender_system_api/bin/python /home/ubuntu/projects/recommender_system_api/manage_dev.py migrate

#permission for logs
sudo chmod -R 0777 /home/ubuntu/projects/recommender_system_api/logs/