---
title: JetBrains IDE Setup
description: 整理 JetBrains IDE 的运行环境变量、代理设置及 CLion 51 单片机开发配置
date: 2023-11-25 12:30:00 +0800
categories: [ Development, Tooling ]
tags: [ JetBrains, CLion, IDE, PlatformIO ]
---

## 通用设置

### 运行与调试环境变量

CLion 和 PyCharm 的环境变量入口相同：进入`运行 -> 编辑配置`，在对应运行配置的`环境变量`中添加项目所需变量。动态库路径、代理等仅对当前运行配置生效的内容，统一在此处维护。

### 代理设置

在运行配置的环境变量中添加：

```text
http_proxy=http://127.0.0.1:7897
https_proxy=http://127.0.0.1:7897
```

![代理环境变量配置](posts/2023-11-25-jetbrains-ide-setup/proxy.png)

## CLion
### 51 单片机开发
#### 安装插件

![Alt text](posts/2023-11-25-jetbrains-ide-setup/plugin.png)  
通过<https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py>
下载`get-platformio.py`文件后使用python运行。随后在`设置->语言和框架`中设置PlatformIO的路径。  
windows和linux稍有不同  
Windows在`C:/Users/UserName/.platformio/penv/Scripts/platformio`  
Linux在`~/.platformio/penv/bin/platformio`

官网链接：<https://docs.platformio.org/en/latest/core/installation/methods/index.html>

#### 配置Shell命令

- Windows在系统环境变量Path附加 C:\Users\UserName\.platformio\penv\Scripts
- Linux可通过在~/.profile中添加

```text
export PATH=$PATH:$HOME/.local/bin
```
{: file="~/.profile" }

然后

```bash
mkdir -p ~/.local/bin
ln -s ~/.platformio/penv/bin/platformio ~/.local/bin/platformio
ln -s ~/.platformio/penv/bin/pio ~/.local/bin/pio
ln -s ~/.platformio/penv/bin/piodebuggdb ~/.local/bin/piodebuggdb
```

或者直接

```bash
sudo mkdir -p /usr/local/bin
sudo ln -s ~/.platformio/penv/bin/platformio /usr/local/bin/platformio
sudo ln -s ~/.platformio/penv/bin/pio /usr/local/bin/pio
sudo ln -s ~/.platformio/penv/bin/piodebuggdb /usr/local/bin/piodebuggdb
```

官网链接：<https://docs.platformio.org/en/latest/core/installation/shell-commands.html#piocore-install-shell-commands>

#### 编写程序

- 选择芯片

![Alt text](posts/2023-11-25-jetbrains-ide-setup/choose.png)  
以`STC89C52`为例  
待环境构建完成后（第一次需下载文件有点慢）  
在CMakeListsPrivate.txt文件中
![Alt text](posts/2023-11-25-jetbrains-ide-setup/basic.png)  
windows在include_directories后添加

```text
include_directories("${ENV_HOME_PATH}/.platformio/packages/toolchain-sdcc/include")
include_directories("${ENV_HOME_PATH}/.platformio/packages/toolchain-sdcc/non-free/include")
```
{: file="CMakeListsPrivate.txt" }

linux在include_directories后添加

```text
include_directories("$ENV{HOME}/.platformio/packages/toolchain-sdcc/share/sdcc/include")
include_directories("$ENV{HOME}/.platformio/packages/toolchain-sdcc/share/sdcc/non-free/include")
```
{: file="CMakeListsPrivate.txt" }

跳转到`8052.h`头文件中，将`#include <8051.h>`改为`#include "8051.h"`并添加`#include "lint.h"`  
![Alt text](posts/2023-11-25-jetbrains-ide-setup/8052.png)  
跳转到`8051.h`头文件中，在开头添加上`#include "lint.h"`  
![Alt text](posts/2023-11-25-jetbrains-ide-setup/8051.png)  
然后可正常编写程序。  
**注意，位定义和中断与keil不同！**

#### 烧录程序

PlatformIO 原生的烧录很慢，改为使用stcgal烧录。  
首先，安装stcgal。新版系统的 Python 不允许直接用 pip 安装到系统环境，这里安装到 PlatformIO 自带的 Python 环境中，并链接到`~/.local/bin`

```bash
~/.platformio/penv/bin/pip install stcgal
ln -s ~/.platformio/penv/bin/stcgal ~/.local/bin/stcgal
```

然后在platformio.ini添加

```text
upload_flags =
    -P
    stc89a
    -p
    $UPLOAD_PORT
upload_command = stcgal $UPLOAD_FLAGS $SOURCE
```
{: file="platformio.ini" }

即可使用stcgal烧录程序。  

#### 附录

stcgal支持的芯片类型如下

|stc89|stc89a|stc12a|stc12b|stc12|stc15a|stc15|stc8|stc8d|stc8g|usb15|
