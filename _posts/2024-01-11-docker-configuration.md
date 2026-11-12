---
title: Docker Configuration
description: 记录 Docker 的免 sudo 权限、守护进程与容器的代理配置、容器内使用 NVIDIA 显卡，以及显示 ROS 图形界面的方法
date: 2024-01-11 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Docker, Linux, Containers ]
series: linux
---

## 权限设置

Docker 进程默认是以 root 用户运行的，免 sudo 运行需要将用户添加到`docker`组中

```bash
# 创建docker组
sudo groupadd docker

# 将用户添加到docker组
sudo usermod -aG docker $USER
```
{: file='添加用户组'}

重新登录（或执行`newgrp docker`）后即可免 sudo 运行 Docker。注意：`docker`组成员等同于拥有 root 权限，只应添加可信用户

## 代理配置

### 守护进程代理

Docker 的守护进程（dockerd）默认不会走代理，需要在进程服务文件夹配置代理设置。该配置只影响 dockerd 本身（如拉取/推送镜像），不会作用于容器内网络和`docker build`中的`RUN`步骤

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
```
{: file='创建进程服务文件夹'}

```bash
sudo vim /etc/systemd/system/docker.service.d/proxy.conf
```
{: file='新建代理文件'}

在代理文件中，添加代理信息，以 clash-verge 为例

```ini
[Service]
Environment="HTTP_PROXY=http://127.0.0.1:7897"
Environment="HTTPS_PROXY=http://127.0.0.1:7897"
Environment="NO_PROXY=localhost,127.0.0.1"
```
{: file='proxy.conf'}

最后重新加载 systemd 配置，并重启 Docker

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```
{: file='重启 Docker'}

```bash
docker info
```

如果输出了代理信息即配置成功

### 容器代理

容器内的程序和`docker build`中的`RUN`步骤使用的代理，在`~/.docker/config.json`中配置，之后新建的容器和构建过程会自动带上对应的环境变量

```json
{
  "proxies": {
    "default": {
      "httpProxy": "http://172.17.0.1:7897",
      "httpsProxy": "http://172.17.0.1:7897",
      "noProxy": "localhost,127.0.0.1"
    }
  }
}
```
{: file='~/.docker/config.json'}

容器内的`127.0.0.1`指向容器自身，访问宿主机上的代理要使用宿主机在`docker0`网桥上的地址（默认为`172.17.0.1`），同时代理软件需要允许来自局域网的连接。使用`--net=host`的容器与宿主机共享网络，可以直接使用`127.0.0.1`

只想对单个容器生效时，可以在启动时通过`-e`传入

```bash
docker run -it -e HTTP_PROXY=http://172.17.0.1:7897 -e HTTPS_PROXY=http://172.17.0.1:7897 ubuntu:24.04
```
{: file='为单个容器设置代理'}

## GPU 支持

容器内使用 NVIDIA 显卡需要安装`nvidia-container-toolkit`。宿主机上要先装好显卡驱动，容器内不需要再安装驱动，CUDA 等运行库由镜像提供

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
{: file='添加软件源'}

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
```
{: file='安装 nvidia-container-toolkit'}

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
{: file='配置 Docker 运行时并重启'}

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```
{: file='验证'}

能输出与宿主机相同的显卡信息即配置成功

```bash
docker run --rm --gpus '"device=0,1"' ubuntu:24.04 nvidia-smi
```
{: file='只使用编号为 0 和 1 的显卡'}

使用`docker compose`时，在服务中声明所需的设备

```yaml
services:
  app:
    image: ubuntu:24.04
    command: nvidia-smi
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```
{: file='compose.yaml'}

## Docker ROS 可视化

需要配置 Docker 以便支持 X11 显示系统，用于图形界面显示。以`osrf`中的`ros:humble-desktop`为例。
在容器外的终端执行如下命令，允许本机的本地连接访问 X server

```bash
xhost +local:
```

不要使用`xhost +`，它会关闭 X server 的访问控制，网络上任何主机都可连接并截屏、记录键盘输入。使用完毕后执行`xhost -local:`恢复

随后启动容器

```bash
docker run -it \
    --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name ros_rviz_container \
    osrf/ros:humble-desktop
```

即可使用`rviz2`等进行可视化显示

参数说明：

| 参数                                          | 说明 |
| :-------------------------------------------- | :--- |
| `--net=host`                                  | 使 Docker 容器共享主机的网络环境，这样 ROS 可以正常通信 |
| `--env="DISPLAY"`                             | 将主机的 DISPLAY 环境变量传递给容器，允许容器中的 GUI 程序使用主机的显示 |
| `--env="QT_X11_NO_MITSHM=1"`                  | 防止共享内存导致的图形渲染问题 |
| `--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"` | 允许容器访问主机的 X11 unix socket |

需要在容器内使用显卡渲染（如`gazebo`、`rviz2`）时，在安装`nvidia-container-toolkit`后加上以下参数

```bash
    --gpus all \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
```
