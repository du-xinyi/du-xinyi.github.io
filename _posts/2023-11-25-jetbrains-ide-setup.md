---
title: JetBrains IDE Setup
description: 整理 JetBrains IDE 的运行环境变量、代理设置，以及 CLion 配合 PlatformIO 开发 51 单片机的配置
date: 2023-11-25 12:30:00 +0800
categories: [ Development, Tooling ]
tags: [ JetBrains, CLion, IDE, PlatformIO ]
---

## 通用设置

### 运行与调试环境变量

CLion 和 PyCharm 的环境变量入口相同：进入`运行 -> 编辑配置`，在对应运行配置的`环境变量`中添加项目所需变量。动态库路径、代理等仅对当前运行配置生效的内容，统一在此处维护

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

![在 CLion 插件市场安装 PlatformIO 插件](posts/2023-11-25-jetbrains-ide-setup/plugin.png)  
通过<https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py>
下载`get-platformio.py`文件后使用 Python 运行。随后在`设置->语言和框架`中设置 PlatformIO 的路径  
Windows 和 Linux 稍有不同  
Windows 在`C:/Users/UserName/.platformio/penv/Scripts/platformio`  
Linux 在`~/.platformio/penv/bin/platformio`

官网链接：<https://docs.platformio.org/en/latest/core/installation/methods/index.html>

#### 配置 Shell 命令

- Windows 在系统环境变量 Path 附加 C:\Users\UserName\.platformio\penv\Scripts
- Linux 可通过在~/.profile 中添加

```text
export PATH=$PATH:$HOME/.local/bin
```
{: file='~/.profile'}

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

![新建工程时选择芯片型号](posts/2023-11-25-jetbrains-ide-setup/choose.png)  
以`STC89C52`为例  
待环境构建完成后（第一次需下载文件有点慢），PlatformIO 会在工程根目录生成`CMakeListsPrivate.txt`，其中包含了宏定义和头文件路径
![PlatformIO 生成的 CMakeListsPrivate.txt](posts/2023-11-25-jetbrains-ide-setup/basic.png)  
其中没有 SDCC 自带头文件的路径，CLion 无法识别`8052.h`等头文件。`CMakeListsPrivate.txt`由 PlatformIO 自动生成，重新初始化工程时会被覆盖，不要直接修改；在工程根目录新建`CMakeListsUser.txt`，该文件会被`CMakeLists.txt`自动引入

Windows 添加

```text
include_directories("${ENV_HOME_PATH}/.platformio/packages/toolchain-sdcc/include")
include_directories("${ENV_HOME_PATH}/.platformio/packages/toolchain-sdcc/non-free/include")
```
{: file='CMakeListsUser.txt'}

Linux 添加

```text
include_directories("$ENV{HOME}/.platformio/packages/toolchain-sdcc/share/sdcc/include")
include_directories("$ENV{HOME}/.platformio/packages/toolchain-sdcc/share/sdcc/non-free/include")
```
{: file='CMakeListsUser.txt'}

跳转到`8052.h`头文件中，将`#include <8051.h>`改为`#include "8051.h"`并添加`#include "lint.h"`  
![修改后的 8052.h](posts/2023-11-25-jetbrains-ide-setup/8052.png)  
跳转到`8051.h`头文件中，在开头添加上`#include "lint.h"`  
![修改后的 8051.h](posts/2023-11-25-jetbrains-ide-setup/8051.png)  
然后可正常编写程序。这两个头文件位于 PlatformIO 的工具链目录中，工具链更新后修改会被还原，需要重新修改  
**注意，位定义和中断与 Keil 不同！**

```c
#include <8052.h>

#define LED P1_0 // Keil中写作 sbit LED = P1^0;

void timer0_isr(void) __interrupt(1) // Keil中写作 void timer0_isr(void) interrupt 1
{
    LED = !LED;
}
```
{: file='SDCC 与 Keil 的写法差异'}

#### 烧录程序

PlatformIO 原生的烧录很慢，改为使用 stcgal 烧录  
首先，安装 stcgal。新版系统的 Python 不允许直接用 pip 安装到系统环境，这里安装到 PlatformIO 自带的 Python 环境中，并链接到`~/.local/bin`

```bash
~/.platformio/penv/bin/pip install stcgal
ln -s ~/.platformio/penv/bin/stcgal ~/.local/bin/stcgal
```

然后在 platformio.ini 添加

```text
upload_flags =
    -P
    stc89a
    -p
    $UPLOAD_PORT
upload_command = stcgal $UPLOAD_FLAGS $SOURCE
```
{: file='platformio.ini'}

即可使用 stcgal 烧录程序  

`-P`后面的参数为芯片使用的协议，需根据芯片型号选择，stcgal 支持的协议如下

```text
stc89  stc89a  stc12a  stc12b  stc12  stc15a  stc15  stc8  stc8d  stc8g  usb15
```

不确定时可以使用`-P auto`自动检测
