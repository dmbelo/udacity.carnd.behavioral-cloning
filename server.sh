#!/bin/bash

# aws ec2 stop-instances --instance-ids 'i-0d67c40ac879ecb9c'

instance_id=i-0d67c40ac879ecb9c

function server_usage
{
  echo "usage: server (start|stop|sync)"
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

function server_sync
{
  echo "Syncing..."
  ip_address=$(aws ec2 describe-instances --instance-ids $instance_id --output text --query 'Reservations[*].Instances[*].PublicIpAddress')
  rsync -avnL --exclude '.git*' --exclude '.DS_Store' . carnd@$ip_address:carnd.behavioral-cloning/
}

case $1 in
  start)
    server_start
    ;;
  stop)
    server_stop
    ;;
  sync)
    server_sync
    ;;
  *)
    server_usage
    exit 1
esac
