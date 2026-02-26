#!/bin/bash
sudo apt-get update
sudo apt-get install ntpdate       # 安装ntpdate
sudo ntpdate time.windows.com      # 更新本地时间

sudo timedatectl set-local-rtc 1 # 将时间更新到硬件
