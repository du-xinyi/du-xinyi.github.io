---
title: Linux Mounting
description: 汇总 Linux 文件系统类型、挂载方式、配置参数与常见操作
date: 2025-09-22 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Linux, Filesystem, Mount ]
---

## 常见文件系统类型

- 内核态本地文件系统
  - ext2/ext3/ext4 → Linux 最常用的文件系统
  - xfs → 高性能文件系统
  - btrfs → 现代化文件系统，支持快照、校验
  - reiserfs → 较旧文件系统，内核 6.13 起已移除
  - vfat / msdos / exfat / ntfs3 → Windows 兼容文件系统（旧的 ntfs 驱动已于内核 6.9 移除，ntfs 现为 ntfs3 的别名）
  - iso9660 → 光盘文件系统
  - zfs → 来自 Solaris，支持快照、压缩，需 ZFS on Linux
- 网络文件系统（内核态）
  - nfs / nfs4 → Network File System
  - cifs / smb3 → Windows 共享（SMB/Samba）
  - afs → Andrew File System
- 用户态挂载（FUSE）
  - fuse.sshfs → SSH 远程文件系统
  - fuse.encfs → 加密文件系统
  - fuse.glusterfs → 分布式文件系统
  - fuseblk → 一般是 NTFS/exFAT 用户态实现（ntfs-3g、exfat-fuse 等）
  - fuse.rclone → 云存储挂载（如 Google Drive、OneDrive 等）
- 伪文件系统（内核虚拟挂载）
  - proc → /proc，进程和内核信息
  - sysfs → /sys，设备和内核信息
  - tmpfs → 内存文件系统
  - devtmpfs → 设备节点文件系统
  - cgroup / cgroup2 → 控制组文件系统
  - securityfs → 内核安全接口（AppArmor、IMA、TPM 等）
  - efivarfs → EFI/UEFI 固件变量
  - debugfs → 内核调试接口
  - tracefs → 内核跟踪接口（ftrace）
  - hugetlbfs → 大页内存文件系统
  - mqueue → POSIX 消息队列
  - configfs → 内核对象配置接口
  - autofs → 自动挂载文件系统
  - binfmt_misc → 额外可执行格式支持
  - fusectl → FUSE 控制接口
  - selinuxfs → SELinux 配置接口
  - rpc_pipefs → RPC 通信管道文件系统
  - nsfs → namespace 文件系统
  - pstore → 内核崩溃日志存储
  - resctrl → 缓存分配与内存带宽控制（Intel RDT / AMD QoS）
- 特殊用途文件系统
  - overlay（OverlayFS）→ 联合文件系统，容器镜像层常用（Docker/Podman）
  - aufs → 旧的联合文件系统，未进入主线内核，Docker 已移除其存储驱动
  - squashfs → 压缩只读文件系统（常见于 LiveCD）
  - ramfs → 原始内存文件系统（比 tmpfs 更基础）

[一键检查脚本](https://raw.githubusercontent.com/du-xinyi/du-xinyi.github.io/master/assets/scripts/check_mounts.sh)

**注意：由于 内核态/用户态切换 和 拷贝开销，用户态挂载（FUSE）性能通常比内核态文件系统差，不推荐在高并发、大吞吐量的场景下直接使用**
