#!/bin/bash

# 安装依赖项
sudo apt install -y build-essential git make libtool pkg-config
sudo apt install -y autoconf automake texinfo libjim-dev
sudo apt install -y libusb-1.0-0-dev libftdi1-dev libhidapi-dev libgpiod-dev libjaylink-dev libcapstone-dev
sudo apt install -y perl

# 编译安装openocd
./bootstrap
./configure
make -j$(nproc)
sudo make install

# 添加udev规则
sudo cp contrib/60-openocd.rules /etc/udev/rules.d/
sudo usermod -aG plugdev $USER

# 更新udev规则
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "已将当前用户加入plugdev组，重新登录后即可免sudo使用openocd"
