---
title: Ubuntu Optimization
description: 汇总 Ubuntu 的语言、内存、时间、显卡及常用系统优化配置
date: 2023-08-29 12:30:00 +0800
permalink: /posts/optimization/
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
| :------: | :-------: |
|  ------  |  -------  |
|   16GB   |   4–8GB   |
|   32GB   |  8–16GB   |
|   64GB   |   16GB    |
|  128GB+  |  16–32GB  |

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
sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="zswap.enabled=1 zswap.compressor=lz4 zswap.max_pool_percent=20 /' /etc/default/grub && \
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

安装`ntpdate`

```bash
sudo apt-get install ntpdate
```

更新本地时间

```bash
sudo ntpdate time.windows.com
```

更新硬件

```bash
sudo timedatectl set-local-rtc 1
```

[一键配置脚本](https://github.com/du-xinyi/du-xinyi.github.io/releases/download/origin/time.sh)

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
### 查询功率限制

```bash
nvidia-smi -q | grep 'Power Limit'
```

![Alt text](posts/2023-08-29-ubuntu-optimization/nvidia-smi.png)  

### 设置持久模式

持久模式是一种让GPU驱动程序在系统启动时就加载并一直保持运行的模式，可以避免在每次运行GPU应用程序时重新初始化GPU的开销

```bash
sudo nvidia-smi -pm 1
```

#### 开机自启

```bash
sudo vim /etc/systemd/system/nvidia-persistenced.service
```

添加如下内容

```bash
[Unit]
Description=NVIDIA Persistence Mode

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pm 1
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```
{: file='nvidia-persistenced.service' }

重新加载 systemd 配置
  
```bash 
sudo systemctl daemon-reload
```

开机自启

```bash 
sudo systemctl enable nvidia-persistenced.service
sudo systemctl start nvidia-persistenced.service
```

### 设置功耗

```bash
sudo nvidia-smi -pl 140
```

#### 开机自启

```bash
sudo vim /etc/systemd/system/nvidia-power.service
```

```bash
[Unit]
Description=NVIDIA Power Restrict

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pl 140
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
```
{: file='nvidia-power.service' }

重新加载 systemd 配置
  
```bash 
sudo systemctl daemon-reload
```

开机自启

```bash 
sudo systemctl enable nvidia-power.service
sudo systemctl start nvidia-power.service
```

使用.run安装的显卡驱动可能不支持此方法修改功耗

### nvidia-powerd

在30系及以后的显卡中，使用了一种新类型的服务nvidia-powerd对显卡功耗进行控制操作

```bash
sudo nvidia-powerd
```

#### 开机自启

```bash
sudo vim /etc/systemd/system/nvidia-powerd.service
```

添加如下内容

```bash
[Unit]
Description=NVIDIA Power Daemon
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/bin/nvidia-powerd

[Install]
WantedBy=multi-user.target
```

重新加载systemd配置

```bash
sudo systemctl daemon-reload
```

启用服务

```bash
sudo systemctl enable nvidia-powerd
```
