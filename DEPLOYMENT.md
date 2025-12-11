# Jivanu RAG - AWS Deployment Guide

## Quick Deploy on AWS EC2

1. Launch EC2: t3.medium, Ubuntu 22.04, 30GB storage
2. SSH: `ssh -i key.pem ubuntu@EC2-IP`
3. Install Docker: `curl -fsSL https://get.docker.com | sh`
4. Clone: `git clone https://github.com/YOUR-USERNAME/jivanu-rag.git`
5. Deploy: `cd jivanu-rag && ./deploy-aws.sh`

## Access
- URL: http://EC2-PUBLIC-IP
- Data persists in: /home/ubuntu/jivanu-rag/data/
