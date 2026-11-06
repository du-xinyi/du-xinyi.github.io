---
title: JetBrains IDE Setup
description: 整理 CLion 与 PyCharm 的嵌入式、ROS、OpenCV 和代理环境配置
date: 2023-11-25 12:30:00 +0800
categories: [ Development, Tooling ]
tags: [ JetBrains, CLion, PyCharm, IDE, PlatformIO, ROS ]
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

### ROS 2 开发
#### 启动CLion

```bash
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja
```

{: file='编译ROS 2 工作空间'}
此时在build目录下会生成文件`compile_commands.json`，然后

```bash
source install/setup.zsh
clion .
```
{: file='启动 CLion'}

在`文件->打开`选择工作空间下`build`文件夹下`compile_commands.json`。作为项目打开
![Alt text](posts/2023-11-25-jetbrains-ide-setup/compile_commands.png)

然后在`工具->编译数据库->更改项目根`选择工作空间作为根目录
![Alt text](posts/2023-11-25-jetbrains-ide-setup/workspace.png)

#### 编译工程

在`文件->设置->工具->外部工具`点击`+`号，新增一个工具。具体填写内容就是之前编译ROS 2 工作空间的命令。程序填写`colcon`
，实参填写`build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja`，工作目录选择ros2的工作空间。
![Alt text](posts/2023-11-25-jetbrains-ide-setup/external_tools.png)

保存后在`工具->外部工具`选择`colcon build`，此时可以编译整个工作空间。

#### 编译调试功能包

1、创建命令脚本  
在功能包下新建一个文件命名为`cmake_commands.bat`，填写如下内容

```bash
/usr/bin/cmake /home/duxinyi/Code/ROS/ROS2/HelloWorld/src/cpp_pubsub -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -G Ninja -DCMAKE_INSTALL_PREFIX=/home/duxinyi/Code/ROS/ROS2/HelloWorld/install/cpp_pubsub
/usr/bin/cmake --build /home/duxinyi/Code/ROS/ROS2/HelloWorld/build/cpp_pubsub -- -j8 -l8
/usr/bin/cmake --install /home/duxinyi/Code/ROS/ROS2/HelloWorld/build/cpp_pubsub
```
{: file='cmake_commands.bat'}

第一行为构建命令，第一个路径为功能包路径，第二个路径为功能包的安装目录；第二行为编译命令，路径为功能包的构建目录；第三行为安装命令，路径为功能包的构建目录。

2、创建自定义构建目标  
在`文件->设置->构建、执行、部署->自定义构建目标`点击`+`号，新增一个目标。在`构建`
中添加外部工具，选择功能包下的`cmake_commands.bat`，并将工作目录选择为功能包的构建目录
![Alt text](posts/2023-11-25-jetbrains-ide-setup/custom_build_target.png)

3、配置run/debug功能  
在`添加配置->编辑配置`点击`+`号，选择自定义构建应用程序，在`目标`选择上一步生成的构建目标，在`可执行文件`
选择功能包编译出的可执行文件，删除`执行前`中的构建。
![Alt text](posts/2023-11-25-jetbrains-ide-setup/custom_build_application.png)

4、断点调试  
在功能包的CMakeLists.txt中添加`SET(CMAKE_BUILD_TYPE "Debug")`

## PyCharm

### ROS 开发

PyCharm 在设置项目解释器后默认是没有添加ROS 环境，在`文件->设置->项目->Python解释器`中，选择全部显示

![Alt text](posts/2023-11-25-jetbrains-ide-setup/python.png)

选择最右侧的`查看解释器路径`

![Alt text](posts/2023-11-25-jetbrains-ide-setup/site.png)

由此可以看出，在PyCharm中没有包含ROS 库，将其添加进解释器路径即可进行ROS 开发

后续使用自定义消息时，同样需要将工作空间内`install/<功能包名>/lib/python3.x/site-packages`添加进解释器路径

#### OpenCV 冲突

ROS 安装的 `cv_bridge` 可能与 pip 安装的 OpenCV 环境发生冲突，导致相关代码运行异常。在通用设置所述的运行配置环境变量中添加：

```text
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ros/humble/lib
```

![OpenCV 环境变量配置](posts/2023-11-25-jetbrains-ide-setup/opencv.png)
