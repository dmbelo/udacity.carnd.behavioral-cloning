#!/bin/bash

# aws ec2 stop-instances --instance-ids 'i-0d67c40ac879ecb9c'

instance_id=i-0d67c40ac879ecb9c

function server_usage
{
  echo "usage: server (start|stop|push|fetch|ssh)"
}

function server_start
{
  echo "Starting..."
  aws ec2 start-instances --instance-ids $instance_id
}

function server_stop
{
  echo "Stopping..."
  aws ec2 stop-instances --instance-ids $instance_id
}

function server_ssh
{
  echo "Starting ssh session..."
  ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --output text --query 'Reservations[*].Instances[*].PublicIpAddress')
  ssh carnd@$ip_address
}

function server_push
{
  echo "Pushing..."
  ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --output text --query 'Reservations[*].Instances[*].PublicIpAddress')
  rsync -avL --exclude-from 'exclude-list.txt' . carnd@$ip_address:carnd.behavioral-cloning/
}

function server_fetch
{
  echo "Fetching..."
  ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --output text --query 'Reservations[*].Instances[*].PublicIpAddress')
  rsync -avuL --exclude-from 'exclude-list.txt' carnd@$ip_address:carnd.behavioral-cloning/ .
}


case $1 in
  start)
    server_start
    ;;
  stop)
    server_stop
    ;;
  push)
    server_push
    ;;
  fetch)
    server_fetch
    ;;
  ssh)
    server_ssh
    ;;
  *)
    server_usage
    exit 1
esac
