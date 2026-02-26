#!/bin/bash

# 安装依赖项
sudo apt install -y make libtool pkg-config
sudo apt install -y autoconf automake texinfo
sudo apt install -y libusb-dev libftdi-dev libhidapi-dev libgpiod-dev libjaylink-dev libcapstone-dev
sudo apt install -y perl python3-ply

# 编译安装openocd
./bootstrap
./configure
make
sudo make install

# 添加udev规则
sudo cp contrib/60-openocd.rules /etc/udev/rules.d/

# 更新udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger