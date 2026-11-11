---
title: OpenOCD Debugging
description: 记录 OpenOCD 的源码编译、udev 规则配置，以及连接芯片、烧录程序和 GDB 调试的方法
date: 2023-10-26 20:30:00 +0800
categories: [ Embedded, Debugging ]
tags: [ OpenOCD, Embedded, Debugging ]
---

`OpenOCD`是一个用来调试嵌入式 SOC 的软件，需要搭配`debug adapter`（如 J-Link，ST-Link，DAP-Link）和 GDB（或 Telnet）一起使用

![GDB、Telnet 通过 OpenOCD 和调试适配器连接芯片的结构](posts/2023-10-26-openocd-debugging/architecture.png)

## Linux 安装 OpenOCD

`apt`仓库中的`openocd`更新较慢，需要较新的芯片或调试器支持时，选择从源码编译

```bash
git clone https://github.com/openocd-org/openocd.git
cd openocd
```
{: file='下载源码'}

### 相关环境安装

#### 基础环境

```bash
sudo apt install -y build-essential git make libtool pkg-config
sudo apt install -y autoconf automake texinfo libjim-dev
```
{: file='基础环境'}

#### USB 适配器驱动

```bash
sudo apt install -y libusb-1.0-0-dev
```
{: file='USB 适配器驱动'}

#### USB-Blaster、ASIX Presto、OpenJTAG 驱动

```bash
sudo apt install -y libftdi1-dev
```
{: file='USB-Blaster、ASIX Presto、OpenJTAG 驱动'}

#### CMSIS-DAP 驱动

```bash
sudo apt install -y libhidapi-dev
```
{: file='CMSIS-DAP 驱动'}

#### linuxgpiod 驱动

```bash
sudo apt install -y libgpiod-dev
```
{: file='linuxgpiod 驱动'}

#### J-Link 驱动

```bash
sudo apt install -y libjaylink-dev
```
{: file='J-Link 驱动'}

#### 额外环境

```bash
sudo apt install -y libcapstone-dev perl
```
{: file='额外环境'}

### 编译源码

在`openocd`源码目录，执行以下命令配置软件包

```bash
./bootstrap
./configure
```
{: file='配置软件包'}

如果全套安装的话，执行完`./configure`最后会显示如下内容  
![configure 输出的调试适配器支持列表](posts/2023-10-26-openocd-debugging/configuration.png)

缺少相关驱动的话相关`debug adapter`的后面则是`no`，最后

```bash
make -j$(nproc)
sudo make install
```
{: file='安装 OpenOCD'}

此时使用`openocd -v`则可查看`openocd`的版本

![openocd -v 输出的版本信息](posts/2023-10-26-openocd-debugging/openocd_version.png)

但是还没完，此时的`openocd`需要`sudo`才能使用，如果要免`sudo`的话需要额外添加`udev`规则

### 添加 udev 规则

在`openocd`源码目录下，将自带的`contrib/60-openocd.rules`复制到`/etc/udev/rules.d`路径下，并将当前用户加入`plugdev`组（需重新登录后生效）

```bash
sudo cp contrib/60-openocd.rules /etc/udev/rules.d/
sudo usermod -aG plugdev $USER
```
{: file='添加 udev 规则'}

最后更新`udev`规则

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```
{: file='更新 udev 规则'}

此时`openocd`可以免`sudo`连接单片机

[一键安装及配置脚本](https://raw.githubusercontent.com/du-xinyi/du-xinyi.github.io/master/assets/scripts/openocd.sh)
，将脚本置于`openocd`源码目录下即可使用

## 连接芯片

`openocd`通过`-f`指定配置文件，至少需要一个调试器配置（`interface`）和一个芯片配置（`target`），自带的配置文件位于`/usr/local/share/openocd/scripts`

```bash
ls /usr/local/share/openocd/scripts/interface
ls /usr/local/share/openocd/scripts/target
```
{: file='查看支持的调试器与芯片'}

以 STM32F1 为例

```bash
openocd -f interface/stlink.cfg -f target/stm32f1x.cfg
```
{: file='使用 ST-Link'}

```bash
openocd -f interface/cmsis-dap.cfg -f target/stm32f1x.cfg
```
{: file='使用 DAP-Link'}

```bash
openocd -f interface/jlink.cfg -c "transport select swd" -f target/stm32f1x.cfg
```
{: file='使用 J-Link（SWD 接口）'}

连接成功后`openocd`保持运行，并监听以下端口

| 端口 | 用途          |
| :--- | :------------ |
| 3333 | GDB 调试      |
| 4444 | Telnet 命令行 |
| 6666 | Tcl 远程调用  |

固定使用的参数可以写入工程目录下的`openocd.cfg`，之后直接执行`openocd`即可

```text
source [find interface/stlink.cfg]
source [find target/stm32f1x.cfg]
```
{: file='openocd.cfg'}

## 烧录程序

```bash
openocd -f interface/stlink.cfg -f target/stm32f1x.cfg \
    -c "program build/firmware.elf verify reset exit"
```
{: file='烧录、校验、复位后退出'}

烧录`bin`文件时需要指定起始地址，如`program build/firmware.bin 0x08000000 verify reset exit`

## GDB 调试

保持`openocd`运行，另开一个终端启动 GDB，并加载带调试信息的`elf`文件

```bash
sudo apt install -y gdb-multiarch
gdb-multiarch build/firmware.elf
```
{: file='启动 GDB'}

```text
(gdb) target extended-remote :3333
(gdb) monitor reset halt
(gdb) load
(gdb) break main
(gdb) continue
```
{: file='连接 OpenOCD、烧录并运行到 main'}

`monitor`后面的内容会原样交给`openocd`执行，常用命令如下

| 命令                 | 作用                     |
| :------------------- | :----------------------- |
| `monitor reset halt` | 复位芯片并停在第一条指令 |
| `monitor reset run`  | 复位芯片并运行           |
| `monitor halt`       | 暂停运行                 |
| `monitor resume`     | 继续运行                 |

## Telnet 命令行

不使用 GDB 时，也可以通过 Telnet 直接向`openocd`发送命令

```bash
telnet localhost 4444
```
{: file='连接 Telnet 命令行'}

| 命令                                   | 作用                         |
| :------------------------------------- | :--------------------------- |
| `reset halt`                           | 复位芯片并停在第一条指令     |
| `halt` / `resume`                      | 暂停 / 继续运行              |
| `reg`                                  | 查看寄存器                   |
| `mdw 0x08000000 16`                    | 从指定地址读取 16 个 32 位字 |
| `flash write_image erase firmware.elf` | 擦除并烧录程序               |
| `shutdown`                             | 关闭`openocd`                |
