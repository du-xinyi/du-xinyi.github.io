---
title: Linux Tools
description: 整理 Linux 下 ssh、scp、rsync、screen、tmux、tar、grep、find、shc、wine 等常用工具的安装与用法
date: 2025-10-31 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Linux, CLI, SSH ]
---

## ssh

`ssh`命令用于通过`SSH`协议连接到远程主机，实现远程登录和执行命令

### 配置 ssh

```bash
sudo apt-get install -y ssh
```
{: file='安装 ssh'}

`ssh`元包会同时安装`openssh-client`和`openssh-server`，安装后`sshd`会监听 22 端口；如果只需要客户端，安装`openssh-client`即可

```bash
ssh-keygen -t ed25519 -C "email@example.com"
```
{: file='生成 ssh key'}

| 参数 | 说明 |
| :--- | :--- |
| `-t` | 指定算法，包括`ed25519`、`ecdsa`、`rsa`以及配合硬件密钥使用的`ed25519-sk`、`ecdsa-sk`，推荐使用`ed25519` |
| `-C` | 注释，用于区分不同的密钥，通常填写邮箱地址 |
| `-b` | 指定密钥长度，`ed25519`长度固定，不需要指定；使用`rsa`时应不小于 3072 |

密钥默认保存在`~/.ssh/id_ed25519`（私钥）和`~/.ssh/id_ed25519.pub`（公钥）。私钥不要泄露或复制给他人，生成时可以设置口令加以保护

```bash
ssh-add ~/.ssh/id_ed25519
```
{: file='添加到 ssh 代理'}

添加到`ssh代理`后，私钥口令只需输入一次（需`ssh-agent`已在运行）；免密登录远程主机还需将公钥复制到远程主机，见下方`ssh-copy-id`

```bash
ssh-add -l
```
{: file='查看相应的 ssh 密钥是否被加载'}

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@remote-host
```
{: file='把公钥复制到远程主机'}

如果`ssh-copy-id`不可用，可以手动追加公钥到远程`~/.ssh/authorized_keys`

### 远程连接

```bash
ssh [options] [user]@[hostname] [command]
```
{: file='基本语法'}

参数说明

| 参数         | 说明                                |
| :----------- | :---------------------------------- |
| `-l`         | 指定登录的用户                      |
| `-p`         | 指定连接到远程主机的端口号，默认 22 |
| `-v`         | 显示调试信息                        |
| `-X`         | 启用 X11 转发                       |
| `[user]`     | 远程主机的用户名                    |
| `[hostname]` | 远程主机的主机名或 IP 地址          |
| `[command]`  | 在远程主机上执行的命令              |

### 配置文件

经常连接的主机可以写入`~/.ssh/config`，之后使用别名连接，`scp`、`rsync`、`git`等基于`ssh`的工具也可以使用该别名

```text
Host myserver
    HostName 192.168.1.10
    User user
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
```
{: file='~/.ssh/config'}

```bash
ssh myserver
```
{: file='使用别名连接'}

## scp

`scp`基于`ssh`，用于在本地与远端之间复制文件或目录

```bash
scp [options] [source] [user@hostname]:[target]
```
{: file='基本语法'}

参数说明

| 参数         | 说明                               |
| :----------- | :--------------------------------- |
| `-r`         | 递归复制整个目录                   |
| `-P`         | 指定远程主机的端口号，默认 22      |
| `-p`         | 保留文件的修改时间、访问时间和权限 |
| `-v`         | 显示的调试信息                     |
| `-C`         | 启用压缩                           |
| `-l`         | 限定使用的带宽，以`Kbit`为单位     |
| `[source]`   | 复制的文件                         |
| `[user]`     | 远程主机的用户名                   |
| `[hostname]` | 远程主机的主机名或 IP 地址         |
| `[target]`   | 远程主机的路径                     |

```bash
scp /path/to/localfile user@remote:/path/to/dest/
```
{: file='将本地文件复制到远程'}

```bash
scp user@remote:/path/to/remotefile /local/dest/
```
{: file='从远程复制到本地'}

```bash
scp -r -P 2222 project_dir user@remote:/home/user/
```
{: file='递归复制目录，指定端口 2222'}

## rsync

`rsync`用于同步文件和目录，只传输发生变化的部分，中断后可以续传，适合大量文件或需要反复同步的场景

```bash
sudo apt-get install -y rsync
```
{: file='安装 rsync'}

```bash
rsync [options] [source] [user@hostname]:[target]
```
{: file='基本语法'}

参数说明

| 参数        | 说明                                               |
| :---------- | :------------------------------------------------- |
| `-a`        | 归档模式，递归复制并保留权限、时间、符号链接等属性 |
| `-v`        | 显示详细信息                                       |
| `-z`        | 传输时压缩                                         |
| `-P`        | 显示进度，并保留未传完的文件以便续传               |
| `--delete`  | 删除目标中存在而源中不存在的文件，使两端完全一致   |
| `--exclude` | 排除匹配的文件或目录                               |
| `-n`        | 只显示将要执行的操作，不实际传输                   |
| `-e`        | 指定远程连接使用的命令，如`-e "ssh -p 2222"`       |

```bash
rsync -avzP project_dir/ user@remote:/home/user/project_dir/
```
{: file='把本地目录同步到远程'}

```bash
rsync -avzP --delete --exclude ".git" project_dir/ user@remote:/home/user/project_dir/
```
{: file='镜像同步并排除 .git 目录'}

源路径末尾带`/`表示同步目录中的内容，不带`/`表示把目录本身复制到目标之下。`--delete`会删除目标端的文件，使用前先加`-n`确认

## screen

`screen`是终端复用工具，能够在一个终端窗口中创建多个虚拟终端会话，并且在`ssh`断开连接或关闭终端后保持会话继续运行

```bash
sudo apt-get install -y screen
```
{: file='安装 screen'}

### 启动会话

```bash
screen
```
{: file='启动新的 screen 会话（自动生成名称）'}

```bash
screen -S session_name
```
{: file='启动终端名字为 session_name 的 screen 会话'}

常用快捷键，先按下`Ctrl`+`a`，松开后再按对应的键

| 快捷键         | 功能                                       |
| :------------- | :----------------------------------------- |
| `Ctrl`+`a` `c` | 创建一个新的`screen`窗口                   |
| `Ctrl`+`a` `n` | 切换到下一个`screen`窗口                   |
| `Ctrl`+`a` `p` | 切换到上一个`screen`窗口                   |
| `Ctrl`+`a` `"` | 列出所有的`screen`窗口                     |
| `Ctrl`+`a` `d` | 分离当前的`screen`会话，会话在后台继续运行 |

### 重连会话

```bash
screen -ls
```
{: file='列出所有正在运行的会话'}

```bash
screen -r session_name
```
{: file='重连到名字为 session_name 的会话'}

```bash
screen -r -d session_name
```
{: file='强制重连到名字为 session_name 的会话'}

```bash
screen -x session_name
```
{: file='共享会话'}

## tmux

`tmux`与`screen`用途相同，另外支持把一个窗口分割成多个窗格

```bash
sudo apt-get install -y tmux
```
{: file='安装 tmux'}

```bash
tmux new -s session_name
```
{: file='新建名字为 session_name 的会话'}

```bash
tmux ls
```
{: file='列出所有会话'}

```bash
tmux attach -t session_name
```
{: file='重连到名字为 session_name 的会话'}

```bash
tmux kill-session -t session_name
```
{: file='关闭会话'}

常用快捷键，先按下`Ctrl`+`b`，松开后再按对应的键

| 快捷键              | 功能                                |
| :------------------ | :---------------------------------- |
| `Ctrl`+`b` `d`      | 分离当前会话，会话在后台继续运行    |
| `Ctrl`+`b` `c`      | 创建一个新窗口                      |
| `Ctrl`+`b` `n`      | 切换到下一个窗口                    |
| `Ctrl`+`b` `p`      | 切换到上一个窗口                    |
| `Ctrl`+`b` `%`      | 左右分割窗格                        |
| `Ctrl`+`b` `"`      | 上下分割窗格                        |
| `Ctrl`+`b` `方向键` | 在窗格之间切换                      |
| `Ctrl`+`b` `x`      | 关闭当前窗格                        |
| `Ctrl`+`b` `[`      | 进入滚动模式查看历史输出，按`q`退出 |

## tar

`tar`用于打包和解包，配合压缩算法生成`.tar.gz`、`.tar.xz`等文件

```bash
tar [options] [archive] [files]
```
{: file='基本语法'}

参数说明

| 参数        | 说明                                        |
| :---------- | :------------------------------------------ |
| `-c`        | 创建归档                                    |
| `-x`        | 解包                                        |
| `-t`        | 列出归档中的内容                            |
| `-f`        | 指定归档文件名，必须紧跟文件名              |
| `-v`        | 显示处理的文件                              |
| `-z`        | 使用`gzip`压缩（`.tar.gz`）                 |
| `-j`        | 使用`bzip2`压缩（`.tar.bz2`）               |
| `-J`        | 使用`xz`压缩（`.tar.xz`），压缩率高但速度慢 |
| `--zstd`    | 使用`zstd`压缩（`.tar.zst`），速度快        |
| `-a`        | 根据归档文件的扩展名自动选择压缩方式        |
| `-C`        | 指定解包到的目录                            |
| `--exclude` | 排除匹配的文件或目录                        |

```bash
tar -czvf project.tar.gz project_dir/
```
{: file='打包并压缩目录'}

```bash
tar -czvf project.tar.gz --exclude="build" --exclude=".git" project_dir/
```
{: file='打包时排除 build 和 .git 目录'}

```bash
tar -xzvf project.tar.gz -C /path/to/dest/
```
{: file='解压到指定目录'}

```bash
tar -tzvf project.tar.gz
```
{: file='查看压缩包内容'}

解包时可以省略压缩算法的选项，`tar -xvf`会自动识别

```bash
tar --zstd -cvf project.tar.zst project_dir/
tar -cavf project.tar.zst project_dir/
```
{: file='使用 zstd 压缩，或由扩展名自动选择'}

### 格式对比

| 格式       | 压缩速度 | 解压速度 | 压缩率              | 兼容性                               |
| :--------- | :------- | :------- | :------------------ | :----------------------------------- |
| `.tar`     | 极快     | 极快     | 无，只打包不压缩    | 极高                                 |
| `.tar.gz`  | 中等     | 快       | 中等                | 极高                                 |
| `.tar.bz2` | 慢       | 慢       | 较高                | 高                                   |
| `.tar.xz`  | 很慢     | 快       | 很高                | 高                                   |
| `.tar.zst` | 极快     | 极快     | 介于 gz 与 bz2 之间 | 较高，需要 tar 1.31 以上并安装`zstd` |

下面是对一个 96MB、以文本为主的目录实测的结果，均使用默认压缩级别，每项重复 3 次取最小值。测量所用的机器为 Intel Core Ultra 9 285K、GeForce RTX 5090 D、64 GB 内存。具体数值与数据内容和机器有关，但相对关系基本如此

| 格式       | 压缩耗时 | 解压耗时 | 压缩后大小 |
| :--------- | :------- | :------- | :--------- |
| `.tar.gz`  | 1.08 秒  | 0.25 秒  | 14.1 MB    |
| `.tar.bz2` | 2.59 秒  | 0.84 秒  | 10.4 MB    |
| `.tar.xz`  | 9.36 秒  | 0.37 秒  | 7.6 MB     |
| `.tar.zst` | 0.14 秒  | 0.17 秒  | 11.3 MB    |

- 需要发给别人或在旧系统上解压时使用`.tar.gz`，几乎所有环境都能直接打开
- 自己备份、传输大文件时使用`.tar.zst`，速度明显快于其他格式
- 只压缩一次、需要长期保存或分发时使用`.tar.xz`，压缩很慢但文件最小，解压并不慢
- `.tar.bz2`在各方面都没有优势，一般只在解压别人提供的文件时遇到

```bash
zip -r project.zip project_dir/
unzip project.zip -d /path/to/dest/
```
{: file='zip 格式的压缩与解压'}

## grep

`grep`用于在文件内容中搜索匹配的行

```bash
grep [options] pattern [files]
```
{: file='基本语法'}

参数说明

| 参数                      | 说明                                  |
| :------------------------ | :------------------------------------ |
| `-r`                      | 递归搜索目录                          |
| `-n`                      | 显示行号                              |
| `-i`                      | 忽略大小写                            |
| `-v`                      | 显示不匹配的行                        |
| `-w`                      | 整词匹配                              |
| `-l`                      | 只显示包含匹配内容的文件名            |
| `-c`                      | 只显示匹配的行数                      |
| `-E`                      | 使用扩展正则表达式                    |
| `-A n` / `-B n` / `-C n`  | 同时显示匹配行之后、之前、前后各`n`行 |
| `--include` / `--exclude` | 只搜索、不搜索匹配的文件              |

```bash
grep -rn "TODO" src/
```
{: file='在目录中递归搜索并显示行号'}

```bash
grep -rn --include="*.cpp" --include="*.h" "main" .
```
{: file='只在指定类型的文件中搜索'}

```bash
grep -E "error|warning" build.log
```
{: file='匹配多个关键字中的任意一个'}

```bash
ps aux | grep python
```
{: file='过滤其他命令的输出'}

## find

`find`用于按文件名、类型、大小、时间等条件查找文件

```bash
find [path] [expression]
```
{: file='基本语法'}

参数说明

| 参数        | 说明                                                 |
| :---------- | :--------------------------------------------------- |
| `-name`     | 按文件名匹配，支持通配符，`-iname`忽略大小写         |
| `-type`     | 按类型匹配，`f`为普通文件，`d`为目录，`l`为符号链接  |
| `-size`     | 按大小匹配，如`+100M`为大于 100MB，`-1k`为小于 1KB   |
| `-mtime`    | 按修改时间匹配，如`+7`为 7 天之前，`-1`为 1 天之内   |
| `-maxdepth` | 限制查找的目录深度                                   |
| `-exec`     | 对找到的每个文件执行命令，`{}`代表文件名，以`\;`结尾 |
| `-delete`   | 删除找到的文件                                       |

```bash
find . -name "*.log"
```
{: file='查找当前目录下所有 .log 文件'}

```bash
find /var/log -type f -size +100M
```
{: file='查找大于 100MB 的文件'}

```bash
find . -type f -mtime +7 -name "*.tmp"
```
{: file='查找 7 天之前修改的 .tmp 文件'}

```bash
find . -name "*.sh" -exec chmod +x {} \;
```
{: file='为所有 .sh 文件添加执行权限'}

`-delete`和`-exec rm`会直接删除文件，执行前先去掉这部分确认查找结果

## shc

shc 是一个用于加密 shell 脚本的工具，它可以将脚本转换为二进制可执行文件，以保护脚本内容不被轻易查看。注意这只是混淆，脚本内容仍可被还原，不要依赖它保护密码等机密

```bash
sudo apt-get install -y shc
```
{: file='安装 shc'}

```bash
[variables] shc [options] script
```
{: file='基本语法'}

参数说明

| 参数      | 说明 |
| :-------- | :--- |
| `-f`      | 指定要加密的脚本文件名。必须提供 |
| `-v`      | 详细模式，输出更多的编译信息 |
| `-r`      | 放松安全限制，生成的二进制文件可以在相同操作系统的不同版本上执行 |
| `-e`      | 设置脚本的过期日期，格式为`dd/mm/yyyy` |
| `-m`      | 设置脚本过期时显示的消息 |
| `-o`      | 指定输出的二进制文件名 |
| `-D`      | 开启调试，输出执行调用信息 |
| `-S`      | 打开`setuid`（使生成程序可被`root`以`setuid`方式调用）。默认关闭 **谨慎使用** |
| `-U`      | 生成不可跟踪的二进制，尝试阻止`strace` / `ptrace` / `truss`等工具跟踪进程 |
| `CC`      | 指定`C`编译器（默认`gcc`） |
| `CFLAGS`  | 传递给编译器的标志，如优化级别或调试信息 |
| `LDFLAGS` | 链接器标志，如静态链接所需标志 |

```bash
shc -f script.sh
```
{: file='最简单生成'}

```bash
shc -r -f script.sh
```
{: file='跨系统生成'}

```bash
shc -f script.sh -o myprog -v
```
{: file='指定输出、启用 verbose'}

```bash
CC=gcc CFLAGS="-O2 -pipe" LDFLAGS="-static" shc -f script.sh -o script_static
```
{: file='使用自定义编译器与静态链接'}

## wine

`Wine`是一个能够在多种`POSIX-compliant`操作系统（诸如`Linux`，`macOS`及`BSD`等）上运行`Windows`应用的兼容层

### 安装

```bash
sudo mkdir -pm755 /etc/apt/keyrings
wget -O - https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -
```
{: file='下载并添加存储库密钥'}

```bash
sudo dpkg --add-architecture i386
```
{: file='启用 32 位存储库'}

```bash
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/$(lsb_release -c | awk '{print $2}')/winehq-$(lsb_release -c | awk '{print $2}').sources
```
{: file='添加源文件'}

```bash
sudo apt update
sudo apt install --install-recommends winehq-stable
```
{: file='安装 Wine'}

### 配置

```bash
sudo apt install winetricks
```
{: file='安装 winetricks'}

#### 补全缺失字体

```bash
winetricks corefonts cjkfonts
wineboot --update
```
{: file='补全缺失字体'}

`corefonts`为 Windows 常用的西文字体，`cjkfonts`为中日韩字体，安装后可解决程序界面中文字显示为方框的问题

#### 运行程序

```bash
wine program.exe
```
{: file='运行 Windows 程序'}

```bash
winecfg
```
{: file='打开 Wine 设置，可修改模拟的 Windows 版本、显示缩放等'}

Wine 的配置和安装的程序默认保存在`~/.wine`目录中，删除该目录即可恢复到初始状态
