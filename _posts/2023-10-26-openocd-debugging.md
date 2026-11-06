---
title: OpenOCD Debugging
description: 记录 OpenOCD 的源码编译、调试适配器配置及嵌入式调试方法
date: 2023-10-26 20:30:00 +0800
categories: [ Embedded, Debugging ]
tags: [ OpenOCD, Embedded, Debugging ]
---

# OpenOCD相关笔记

`OpenOCD`是一个用来调试嵌入式SOC的软件，需要搭配`debug adapter`（如JLink，ST-Link，DAP-Link）和GDB（或Telnet）一起使用

## Linux安装OpenOCD

直接使用`apt`安装的`openocd`
版本很低，直接会导致大部分芯片无法使用，还是选择从源码编译，从[openocd](https://github.com/openocd-org/openocd.git)下载源码

### 相关环境安装

#### 基础环境

```bash
sudo apt install -y build-essential git make libtool pkg-config
sudo apt install -y autoconf automake texinfo libjim-dev
```
{: file='基础环境'}

#### USB适配器驱动

```bash
sudo apt install -y libusb-1.0-0-dev
```
{: file='USB适配器驱动'}

#### USB-Blaster、ASIX Presto、OpenJTAG驱动

```bash
sudo apt install -y libftdi-dev
```
{: file='USB-Blaster、ASIX Presto、OpenJTAG驱动'}

#### CMSIS-DAP驱动

```bash
sudo apt install -y libhidapi-dev
```
{: file='CMSIS-DAP驱动'}

#### linuxgpiod驱动

```bash
sudo apt install -y libgpiod-dev
```
{: file='linuxgpiod驱动'}

#### J-Link驱动

```bash
sudo apt install -y libjaylink-dev
```
{: file='J-Link驱动'}

#### 额外环境

```bash
sudo apt install -y libcapstone-dev perl
```

### 编辑源码

在`openocd`源码目录，执行以下命令配置软件包

```bash
./bootstrap
./configure
```
{: file='配置软件包'}

如果全套安装的话，执行完`./configure`最后会显示如下内容  
![configuration](posts/2023-10-26-openocd-debugging/configuration.png)

缺少相关驱动的话相关`debug adapter`的后面则是`no`，最后

```bash
make
sudo make install
```
{: file='安装OpenOCD'}

此时使用`openocd -v`则可查看`openocd`的版本

![Alt text](posts/2023-10-26-openocd-debugging/openocd_version.png)

但是还没完，此时的`openocd`需要`sudo`才能使用，如果要免`sudo`的话需要额外添加`udev`规则

### 添加udev规则

在`openocd`源码目录下，将自带的`contrib/60-openocd.rules`复制到`/etc/udev/rules.d`路径下，并将当前用户加入`plugdev`组（需重新登录后生效）

```bash
sudo cp contrib/60-openocd.rules /etc/udev/rules.d/
sudo usermod -aG plugdev $USER
```
{: file='添加udev规则'}

最后更新`udev`规则

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
{: file='更新udev规则'}

此时`openocd`可以免`sudo`连接单片机

[一键安装及配置脚本](https://github.com/du-xinyi/du-xinyi.github.io/releases/download/origin/openocd.sh)
，将脚本置于`openocd`源码目录下即可使用
