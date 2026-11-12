---
title: Linux Mounting
description: 汇总 Linux 文件系统类型、挂载与卸载命令、挂载选项及 fstab 配置
date: 2025-09-22 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Linux, Filesystem, Mount ]
series: linux
---

## 常见文件系统类型

- 内核态本地文件系统
  - ext2/ext3/ext4 → Linux 最常用的文件系统
  - xfs → 高性能文件系统
  - btrfs → 现代化文件系统，支持快照、校验
  - f2fs → 面向闪存设备的文件系统
  - vfat / exfat / ntfs3 → Windows 兼容文件系统
  - iso9660 → 光盘文件系统
  - zfs → 支持快照、压缩，需额外安装 OpenZFS
- 网络文件系统（内核态）
  - nfs / nfs4 → Network File System
  - cifs / smb3 → Windows 共享（SMB/Samba）
- 用户态挂载（FUSE）
  - fuse.sshfs → SSH 远程文件系统
  - fuse.rclone → 云存储挂载（如 Google Drive、OneDrive 等）
  - fuse.glusterfs → 分布式文件系统
  - fuseblk → 块设备上的用户态实现（ntfs-3g、exfat-fuse 等）
- 伪文件系统（内核虚拟挂载）
  - proc → /proc，进程和内核信息
  - sysfs → /sys，设备和内核信息
  - tmpfs → 内存文件系统
  - devtmpfs → 设备节点文件系统
  - cgroup2 → 控制组文件系统
  - securityfs → 内核安全接口（AppArmor、IMA、TPM 等）
  - efivarfs → EFI/UEFI 固件变量
  - debugfs / tracefs → 内核调试与跟踪接口
  - hugetlbfs → 大页内存文件系统
  - mqueue → POSIX 消息队列
  - configfs → 内核对象配置接口
  - autofs → 自动挂载文件系统
  - binfmt_misc → 额外可执行格式支持
  - fusectl → FUSE 控制接口
  - pstore → 内核崩溃日志存储
- 特殊用途文件系统
  - overlay（OverlayFS）→ 联合文件系统，容器镜像层常用（Docker/Podman）
  - squashfs → 压缩只读文件系统（常见于 LiveCD、snap）
  - ramfs → 原始内存文件系统（比 tmpfs 更基础）

[一键检查脚本](https://raw.githubusercontent.com/du-xinyi/du-xinyi.github.io/master/assets/scripts/check_mounts.sh)，按以上分类列出当前系统的所有挂载点

**注意：由于内核态/用户态切换和拷贝开销，用户态挂载（FUSE）性能通常比内核态文件系统差，不推荐在高并发、大吞吐量的场景下直接使用**

## 查看挂载信息

```bash
lsblk -f
```
{: file='查看块设备、文件系统类型、UUID 和挂载点'}

```bash
findmnt
```
{: file='以树状结构查看所有挂载点'}

```bash
findmnt /mnt/data
```
{: file='查看指定挂载点的设备、类型和挂载选项'}

```bash
df -hT
```
{: file='查看各文件系统的类型和空间占用'}

## 挂载与卸载

Linux 只有一棵目录树。挂载就是把一个文件系统（磁盘分区、网络共享、内存盘等）接到目录树中的某个目录上，这个目录称为挂载点，之后访问该目录就是访问对应的文件系统

![不同的设备和存储挂载到同一棵目录树的不同目录上](posts/2025-09-22-linux-mounting/mount_tree.png)

```bash
mount [-t 类型] [-o 选项] 设备 挂载点
```
{: file='基本语法'}

挂载点必须是已存在的目录，挂载后目录中原有的内容会被遮盖，卸载后恢复。文件系统类型通常可以自动识别，不需要指定`-t`

```bash
sudo mkdir -p /mnt/data
sudo mount /dev/sdb1 /mnt/data
```
{: file='挂载分区'}

```bash
sudo mount -o ro,noexec /dev/sdb1 /mnt/data
```
{: file='只读且禁止执行'}

```bash
sudo mount -o remount,rw /mnt/data
```
{: file='不卸载的情况下修改挂载选项'}

```bash
sudo umount /mnt/data
```
{: file='卸载，参数可以是挂载点或设备'}

卸载时提示`target is busy`，说明仍有进程在使用该挂载点（包括终端的当前目录位于其中）

```bash
sudo lsof +f -- /mnt/data
```
{: file='查看占用挂载点的进程'}

```bash
sudo umount -l /mnt/data
```
{: file='延迟卸载，立即从目录树中摘除，待不再占用时完成卸载'}

### 常用挂载选项

| 选项                  | 作用 |
| :-------------------- | :--- |
| `defaults`            | 默认选项，等价于`rw,suid,dev,exec,auto,nouser,async` |
| `ro` / `rw`           | 只读 / 读写 |
| `noexec`              | 禁止执行其中的程序 |
| `nosuid`              | 忽略 setuid/setgid 位 |
| `nodev`               | 不解析设备文件 |
| `noatime`             | 不更新访问时间，减少写入 |
| `noauto`              | 不随`mount -a`和开机自动挂载 |
| `nofail`              | 设备不存在时不报错，避免开机卡住 |
| `user`                | 允许普通用户挂载 |
| `uid=` / `gid=`       | 指定文件属主和属组，用于 vfat、exfat、ntfs3 等没有 Linux 权限的文件系统 |
| `umask=`              | 指定权限掩码，适用范围同上 |
| `x-systemd.automount` | 首次访问时才挂载 |

### 常见场景

```bash
sudo mount -o uid=$(id -u),gid=$(id -g),umask=022 /dev/sdb1 /mnt/usb
```
{: file='挂载 NTFS/exFAT 分区，并把属主设为当前用户'}

```bash
sudo mount -o loop,ro ubuntu.iso /mnt/iso
```
{: file='挂载 ISO 镜像'}

```bash
sudo mount -t tmpfs -o size=2G tmpfs /mnt/ramdisk
```
{: file='挂载 2GB 的内存盘'}

```bash
sudo mount --bind /data/project /srv/project
```
{: file='把一个目录绑定挂载到另一个位置'}

```bash
sudo apt install -y nfs-common
sudo mount -t nfs 192.168.1.10:/export/data /mnt/nfs
```
{: file='挂载 NFS 共享'}

```bash
sudo apt install -y cifs-utils
sudo mount -t cifs //192.168.1.10/share /mnt/smb -o credentials=/root/.smbcredentials,uid=$(id -u),gid=$(id -g)
```
{: file='挂载 Windows/Samba 共享'}

其中凭据文件内容如下，权限应设为`600`，避免把密码写在命令行或`fstab`中

```text
username=user
password=pass
```
{: file='/root/.smbcredentials'}

```bash
sudo apt install -y sshfs
sshfs user@remote:/home/user /mnt/remote
fusermount3 -u /mnt/remote
```
{: file='通过 SSH 挂载远程目录及卸载'}

## 开机自动挂载

开机自动挂载通过`/etc/fstab`配置，每行一个挂载项，共 6 个字段

```text
<设备> <挂载点> <类型> <选项> <dump> <pass>
```
{: file='/etc/fstab 字段'}

- 设备：推荐使用`UUID=`，`/dev/sdX`这样的设备名在增减硬盘后可能变化
- dump：是否由`dump`工具备份，一般为`0`
- pass：开机时`fsck`检查顺序，根分区为`1`，其他需要检查的分区为`2`，不检查为`0`

```bash
sudo blkid /dev/sdb1
```
{: file='查询分区的 UUID'}

```text
UUID=1234abcd-12ab-34cd-56ef-1234567890ab  /mnt/data  ext4   defaults,nofail            0  2
UUID=ABCD-1234                             /mnt/usb   exfat  defaults,nofail,uid=1000,gid=1000,umask=022  0  0
192.168.1.10:/export/data                  /mnt/nfs   nfs    defaults,nofail,_netdev    0  0
tmpfs                                      /mnt/ramdisk  tmpfs  size=2G                 0  0
```
{: file='/etc/fstab 示例'}

修改后先检查再挂载，确认无误后再重启。`fstab`写错且没有`nofail`时，系统开机会进入紧急模式

```bash
sudo findmnt --verify
sudo systemctl daemon-reload
sudo mount -a
```
{: file='检查语法并挂载 fstab 中的所有项'}

- 非必需的磁盘和网络存储加上`nofail`，设备缺失时系统仍可正常启动
- 网络文件系统加上`_netdev`，等待网络就绪后再挂载
