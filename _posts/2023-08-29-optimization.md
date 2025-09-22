---
title: Optimize
date: 2023-08-29 12:30:00 +0800
categories: [ Notepad, Linux ]
tags: [ Linux, Optimize ]
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

## zsh

众所周知，大部分的Linux发行版的默认命令解释器是bash，Mac默认的是zsh。相比于默认的bash，zsh有更多的自定义选项，并支持扩展。因此zsh可以实现更强大的命令补全，命令高亮等一系列功能。不过代价就是启动速度会比原生的bash慢，这一点在添加了较多环境后尤为明显。  
默认的zsh配置有点麻烦。因此一个叫robbyrussel的用户在GitHub上制作了一个配置文件`oh-my-zsh`，这是目前为止最流行的zsh配置。(
oh-my-zsh项目地址)[https://github.com/robbyrussell/oh-my-zsh]

### 安装zsh

```bash
sudo apt-get install zsh
```

使用`cat /etc/shells`查看系统可以用的shell

```bash
# /etc/shells: valid login shells
/bin/sh
/bin/bash
/usr/bin/bash
/bin/rbash
/usr/bin/rbash
/bin/dash
/usr/bin/dash
/bin/zsh
/usr/bin/zsh
```
{: file='/etc/shells'}

### 设置系统默认shell

```bash
chsh -s /bin/zsh
```

新开一个终端，就可以使用zsh

### 安装oh-my-zsh

```bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

### 配置

zsh的配置在用户当前目录的`.zshrc`里

#### 修改主题

在[Themes](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)
中查看内置的主题样式和对应的主题名。这些内置主题已经放在`～/.oh-my-zsh/themes`目录下，不需要再下载。默认的是`robbyrussell`
，个人习惯用`ys`

#### 安装插件

`oh-my-zsh`已经内置了`git`插件，内置插件可以在`～/.oh-my-zsh/plugins`
中查看，更多插件可以在[awesome-zsh-plugins](https://github.com/unixorn/awesome-zsh-plugins) 里查看。常用的需要额外下载的插件如下

- zsh-autosuggestions  
  `zsh-autosuggestions`是一个命令提示插件，当输入命令时，会自动推测可能需要输入的命令，按下右键可以快速采用建议。  
  安装步骤：
  - 把插件下载到本地的`~/.oh-my-zsh/custom/plugins`目录
    ```bash
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
    ```
    {: file='~/.oh-my-zsh/custom/plugins' }

  - 在`.zshrc`中，把`zsh-autosuggestions`加入插件列表
    ```yaml
    plugins=(
        # other plugins...
        zsh-autosuggestions  # 插件之间使用空格隔开
    )
    ```
    {: file='.zshrc' }

  - 开启新的终端或执行`source ~/.zshrc`

- zsh-syntax-highlighting  
  `zsh-syntax-highlighting`
  是一个命令语法校验插件，在输入命令的过程中，若指令不合法，则指令显示为红色，若指令合法就会显示为绿色。  
  安装步骤：
  - 把插件下载到本地的`~/.oh-my-zsh/custom/plugins`目录
    ```bash
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting 
    ```
    {: file='~/.oh-my-zsh/custom/plugins' }
  - 在`.zshrc`中，把 `zsh-syntax-highlighting` 加入插件列表
    ```yaml
    plugins=(
        # other plugins...
        zsh-syntax-highlighting
    )
    ```
    {: file='.zshrc' }

  - 开启新的终端或执行`source ~/.zshrc`

- extract
  `extract`用于解压任何压缩文件，不必根据压缩文件的后缀名来压缩，使用`x`命令即可解压文件

安装步骤：

- 直接在`.zshrc`中，把 `extract` 加入插件列表
    ```yaml
    plugins=(
        # other plugins...
        extract
    )
    ```
    {: file='.zshrc' }

- conda-zsh-completion  
  `conda-zsh-completion`是conda的自动补全插件
  安装步骤：
  - 把插件下载到本地的`~/.oh-my-zsh/custom/plugins`目录
    ```bash
     git clone https://github.com/esc/conda-zsh-completion $ZSH_CUSTOM/plugins/conda-zsh-completion
    ```
    {: file='~/.oh-my-zsh/custom/plugins' }

  - 在`.zshrc`中，把`conda-zsh-completion`加入插件列表
    ```yaml
    plugins=(
        # other plugins...
        conda-zsh-completion  # 插件之间使用空格隔开
    )
    ```
    {: file='.zshrc' }

  - 开启新的终端或执行`source ~/.zshrc`

#### 额外设置

1. 在`.zshrc`中额外添加

```yaml
setopt nonomatch
```
{: file='.zshrc' }

以实现通配符

### ros2环境配置

在zshrc中source了环境后无法补全ros2和colcon的相关命令，首先安装

```bash
sudo apt-get install python3-argcomplete
```

然后在source环境前加上

```yaml
eval "$(register-python-argcomplete ros2)"
eval "$(register-python-argcomplete colcon)"
```
{: file='.zshrc' }

以实现补全和快速跳转

[一键配置脚本](https://github.com/du-xinyi/du-xinyi.github.io/releases/download/origin/zsh.sh)

## 时间相关
### 双系统时间错误

在Linux中安装`ntpdate`

```bash
sudo apt-get install ntpdate
```

更新本地时间

```bash
sudo ntpdate time.windows.com
```

最后，更新硬件：

```bash
sudo hwclock --localtime --systohc
```

ubuntu24.04需要使用如下方式更新硬件

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

![Alt text](posts/2023-08-29-optimization/nvidia-smi.png)  

### 设置持久模式

持久模式是一种让GPU驱动程序在系统启动时就加载并一直保持运行的模式，可以避免在每次运行GPU应用程序时重新初始化GPU的开销

```bash
sudo nvidia-smi -pm 1
```

#### 开机自启

```bash
sudo vim /lib/systemd/system/nvidia-persistenced.service
```

添加如下内容

```bash
[Unit]
Description=NVIDIA Persistence Mode

[Service]
Type=oneshot
ExecStart=/usr/bin/sudo /usr/bin/nvidia-smi -pm 1
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
sudo vim /lib/systemd/system/nvidia-power.service
```

```bash
[Unit]
Description=NVIDIA Power Restrict

[Service]
Type=oneshot
ExecStart=/usr/bin/sudo /usr/bin/nvidia-smi -pl 140
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

添加`D-Bus`权限

```bash
sudo vim /etc/dbus-1/system.d/nvidia-powerd.conf
```

添加如下内容

```bash
<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
"http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <policy user="root">
    <allow own="nvidia.powerd.server"/>
    <allow send_destination="nvidia.powerd.server"/>
    <allow receive_sender="nvidia.powerd.server"/>
  </policy>
</busconfig>
```

启用服务

```bash
sudo systemctl enable nvidia-powerd
```