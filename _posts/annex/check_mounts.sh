#!/bin/bash

echo "===== Mount Classification ====="
echo

# 使用 findmnt 获取挂载信息（更可靠）
# 输出格式：挂载点  文件系统类型
mounts=$(findmnt -rno TARGET,FSTYPE)

# 分类函数
classify_fs() {
    local fstype="$1"
    case "$fstype" in
        # 本地磁盘文件系统（内核态）
        ext2|ext3|ext4|xfs|btrfs|reiserfs|vfat|msdos|ntfs|ntfs3|iso9660|exfat|zfs)
            echo "Local filesystem"
            ;;
        # 网络文件系统（内核态）
        nfs*|cifs|smbfs|afs)
            echo "Network filesystem"
            ;;
        # 用户态文件系统（FUSE）
        fuse*|fuseblk|fusectl)
            echo "User-space (FUSE) filesystem"
            ;;
        # 虚拟/伪文件系统（内核虚拟挂载）
        proc|sysfs|tmpfs|devtmpfs|devpts|cgroup*|pstore|efivarfs|bpf|hugetlbfs|mqueue|debugfs|tracefs|configfs|autofs|binfmt_misc|securityfs|fusectl|selinuxfs|rpc_pipefs|nsfs|overlay|overlayfs|ramfs|kafsf|kernfs|resctrl)
            echo "Virtual / pseudo filesystem"
            ;;
        # 特殊用途文件系统
        squashfs|aufs|overlay|ramfs)
            echo "Special-purpose filesystem"
            ;;
        *)
            echo "Other/Unknown filesystem"
            ;;
    esac
}

# 统计用关联数组
declare -A stats

# 遍历输出结果
while read -r target fstype; do
    category=$(classify_fs "$fstype")
    printf "%-30s %-15s %-30s\n" "$target" "$fstype" "$category"
    ((stats["$category"]++))
done <<< "$mounts"

# 打印统计结果
echo
echo "===== Statistics ====="
for key in "${!stats[@]}"; do
    printf "%-30s %d\n" "$key" "${stats[$key]}"
done
