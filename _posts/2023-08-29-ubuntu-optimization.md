---
title: Ubuntu Optimization
description: 汇总 Ubuntu 的语言、内存、时间、显卡及常用系统优化配置
date: 2023-08-29 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Ubuntu, Linux ]
---

# ubuntu优化
## 谷歌浏览器汉化

在ubuntu内更改Chrome的语言时，并没有像是windows的`以这种语言显示Google Chrome`
的选项。最多也只能修改提示翻译的优先语言。没有严格意义上的`更改显示语言`的选项。  
在linux系统上，Chrome的显示语言是根据系统语言变化，一种办法是将系统语言设置成想要显示的语言，但是更多的场景可能是系统语言是英文，应用语言是中文。这里需要通过添加命令行参数来修改
Chrome 的语言。

打开终端，在终端中输入

```bash
LANGUAGE=zh_cn google-chrome
```

此时启动Chrome则是以中文显示。若要修改Chrome的默认语言，需要额外添加脚本。

在Chrome的图标文件`/usr/share/applications/google-chrome.desktop`中

```yaml
[ Desktop Entry ]
  Version=1.0
  Name=Google Chrome
  # Only KDE 4 seems to use GenericName, so we reuse the KDE strings.
  # From Ubuntu's language-pack-kde-XX-base packages, version 9.04-20090413.
  GenericName=Web Browser
  ......
  Exec=/usr/bin/google-chrome-stable %U
  StartupNotify=true
  Terminal=false
  Icon=google-chrome
  Type=Application
  Categories=Network;WebBrowser;
  ......
```
{: file="/usr/share/applications/google-chrome.desktop" }

Chrome通过`Exec=/usr/bin/google-chrome-stable`运行，但直接添加命令行参数`LANGUAGE=zh_cn`
的话不仅会无法使用，而且还会导致GNOME桌面找不到Chrome。解决方法如下

新建一个`chrome_cn.sh`脚本，在脚本中添加如下内容

```bash
#!/bin/bash
LANGUAGE=zh_cn /usr/bin/google-chrome-stable $@
```
{: file="chrome_cn.sh" }

随后将该脚本复制到`/usr/bin/`路径下，最后将`/usr/share/applications/google-chrome.desktop`
里的所有`/usr/bin/google-chrome-stable`换成`/usr/bin/chrome_cn.sh`，此时启动Chrome界面将是中文。

[一键配置脚本](https://github.com/du-xinyi/du-xinyi.github.io/releases/download/origin/google-chinesization.sh)

## pip补全

`pip`提供了自动补全命令，但需要手动开启

```bash
pip completion --bash >> ~/.bashrc
```
{: file="bash终端" }

```zsh
pip completion --zsh >> ~/.zshrc
```
{: file="zsh终端" }

执行如上命令，重启终端后`pip`即可进行自动补全

## swap分区扩容

在大内存机器（如 64GB）上，默认安装时通常只会分配较小的 swap（例如 2GB）。  
虽然物理内存足够，但在以下场景中仍然可能触发 OOM（Out Of Memory）：

- 大型 C++ 工程编译（如 OpenCV / ROS 全编译）
- 深度学习训练
- 多进程程序
- Docker 容器叠加
- 数据集加载瞬时暴涨

因此建议配置 适量 swap 作为缓冲区，防止进程被系统直接杀死

### 查看当前 swap 状态

```bash
free -h
swapon --show
```

| 物理内存 | 推荐 swap |
| :------- | --------: |
| ------   |   ------- |
| 16GB     |     4–8GB |
| 32GB     |    8–16GB |
| 64GB     |      16GB |
| 128GB+   |   16–32GB |

### 使用 swapfile 扩容

关闭旧 swap
```bash
sudo swapoff /swap.img
```

删除旧文件
```bash
sudo rm /swap.img
```

创建新的 swap（示例 16GB）
```bash
sudo fallocate -l 16G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
```

如果文件名不为`swap.img`，需要修改`/etc/fstab`，确保如下行正确

```bash
/swap.img none swap sw 0 0
```
{: file="/etc/fstab" }

### 启用 zswap

`zSwap`是Linux内核的一个功能，它为交换页提供了一个压缩的回写缓存，作为一种虚拟内存压缩形式  

开启`zswap`
```bash
echo 1 | sudo tee /sys/module/zswap/parameters/enabled
```

永久生效
```bash
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="zswap.enabled=1 zswap.compressor=lz4 zswap.max_pool_percent=20 /' /etc/default/grub
```

更新设置
```bash
sudo update-grub
```

### sysctl 内核优化

内核优化
```bash
sudo tee /etc/sysctl.d/99-memory-opt.conf > /dev/null << 'EOF'
vm.swappiness=10
vm.overcommit_memory=1
vm.overcommit_ratio=90
vm.vfs_cache_pressure=50
EOF
```

更新设置
```bash
sudo sysctl --system
```

参数详解：
- vm.swappiness  
控制系统使用 swap 的“积极程度”。默认为`60`，数值越小，越倾向使用物理内存。对于开发机器建议调低

  |     场景     | 推荐值 |
  | :----------: | :----: |
  | ------------ | -----  |
  |   桌面开发   |   10   |
  |    服务器    | 10–20  |
  | 内存较小机器 |   30   |

- vm.overcommit_memory  
控制内存分配策略

  |      值      |       含义       |
  | :----------: | :--------------: |
  | ------------ |      -----       |
  |      0       | 默认（内核估算） |
  |      1       |   允许过度分配   |
  |      2       |     严格限制     |

- vm.overcommit_ratio  
`overcommit_memory=2` 时才严格生效，表示可分配内存比例（%）

- vm.vfs_cache_pressure  
控制 inode/dentry 缓存回收速度

  |      值      |    行为    |
  | :----------: | :--------: |
  | ------------ |   -----    |
  |     100      |    默认    |
  |     <100     | 更保留缓存 |
  |     >100     |  更快回收  |

## 时间相关
### 双系统时间错误

Windows 默认将硬件时钟（RTC）解释为本地时间，而 Ubuntu 通常按 UTC 解释。同一个硬件时钟使用不同约定，会导致切换系统后时间偏移，例如在中国标准时间（UTC+8）下相差 8 小时。建议将两个系统的硬件时钟统一为 UTC，桌面仍按各自设置的时区显示当地时间

#### Windows 使用 UTC 硬件时钟

以管理员身份打开命令提示符，执行以下命令（64 位 Windows 同样使用 `REG_DWORD`）

```cmd
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\TimeZoneInformation" /v RealTimeIsUniversal /t REG_DWORD /d 1 /f
```

重启 Windows，并在“设置 → 时间和语言 → 日期和时间”中确认时区正确，启用自动设置时间并同步。

#### Ubuntu 使用 UTC 硬件时钟

先检查时区与同步状态

```bash
timedatectl
```

如果时区不正确，按实际所在地修改，以下以中国标准时间为例

```bash
sudo timedatectl set-timezone Asia/Shanghai
```

使用系统已有的时间同步服务（如 `systemd-timesyncd` 或 `chrony`）自动校时，无需额外安装 `ntpdate`。可通过以下命令启用可用的同步服务

```bash
sudo timedatectl set-ntp true
```

确认系统时间正确且已同步后，将硬件时钟设为 UTC；该命令也会用当前系统时间更新硬件时钟

```bash
sudo timedatectl set-local-rtc 0
timedatectl
```

### 修改启动等待时间为0

```bash
sudo vim /etc/grub.d/30_os-prober
```

修改

```bash
set timeout=0
```

```bash
sudo vim /etc/default/grub
```

修改

```bash
GRUB_TIMEOUT=0
```

最后更新`grub.cfg`文件，使改动生效

```bash
sudo update-grub
```

### 修改系统重启默认等待时间

```bash
sudo vim /etc/systemd/system.conf
```

将`#DefaultTimeoutStopSec=90s`取消注释，并将`DefaultTimeoutStopSec=90s`改为`DefaultTimeoutStopSec=3s`

**千万不要修改`DefaultTimeoutStartSec`，若修改时间太短，时间将不足以支持系统启动。如果不慎修改，在grub引导中选择recovery模式的root终端，用vim将其改回来**

## 显卡相关
### 查询功率信息

查看 GPU 0 的功率信息，包括设备支持时报告的当前、默认及最小/最大功率上限。多显卡机器可先使用 `nvidia-smi -L` 查看设备编号，再替换 `-i 0` 中的编号；不支持的字段可能显示为 `N/A`。

```bash
nvidia-smi -i 0 -q -d POWER
```

![NVIDIA 显卡功率限制查询示例](posts/2023-08-29-ubuntu-optimization/nvidia-smi.png)

### 持久模式（按需启用）

持久模式用于在没有应用程序使用 GPU 时保留其初始化状态，减少后续计算任务的初始化开销，主要适用于无图形界面的计算场景。如果桌面图形服务已持续使用该 GPU，通常无需额外开启。

NVIDIA 推荐使用 `nvidia-persistenced` 守护进程管理持久状态。先检查驱动包是否已经提供服务及其运行状态

```bash
systemctl cat nvidia-persistenced.service
systemctl status nvidia-persistenced.service
```

如果服务存在但未运行，并且确实需要持久模式，可启动已有服务

```bash
sudo systemctl start nvidia-persistenced.service
```

在设备支持时，为 GPU 0 启用持久模式并查看结果

```bash
sudo nvidia-smi -i 0 -pm 1
nvidia-smi -i 0 --query-gpu=persistence_mode --format=csv
```

`nvidia-smi -pm 1` 本身不会保存跨重启配置。开机行为应以驱动包提供的服务及启动参数为准；部分服务由依赖关系自动启动，不能直接用 `systemctl enable` 启用。如果服务不存在，应检查所安装驱动包的说明，不要创建同名的简化服务覆盖官方配置。

### nvidia-powerd（受支持笔记本的 Dynamic Boost）

`nvidia-powerd` 为 Linux 提供 Dynamic Boost 支持，根据负载在 CPU 和 GPU 之间动态分配功率预算

硬件需要同时满足以下条件：

- 笔记本平台
- Ampere 或更新架构的 NVIDIA GPU
- 受支持的 Intel 或 AMD 平台
- 系统 BIOS 支持 Dynamic Boost

具体平台及软件要求应以所安装驱动版本的文档为准。可通过以下命令查询 BIOS 是否报告支持；其中 `*` 匹配各 GPU 的 PCI 地址目录

```bash
cat /proc/driver/nvidia/gpus/*/power
```

查看输出中的 `Dynamic Boost` 支持状态。确认平台支持后，检查驱动包提供的服务

```bash
systemctl cat nvidia-powerd.service
```

如果服务已安装，且驱动要求的 D-Bus 等配置已就绪，可启用并立即启动

```bash
sudo systemctl enable --now nvidia-powerd.service
systemctl status nvidia-powerd.service
```

若服务启动失败，可查看日志排查平台支持及配置问题

```bash
journalctl -u nvidia-powerd.service -b
```

不支持的平台无需启用该服务
