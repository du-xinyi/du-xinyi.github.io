---
title: RealSense
date: 2023-08-31 18:30:00 +0800
categories: [ Notepad, Camera ]
tags: [ Camera, ROS ]
math: true
---

# RealSense环境配置
## 安装librealsense

参考librealsense的github官方教程[distribution_linux](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

### 下载librealsense源码

```bash
git clone https://github.com/IntelRealSense/librealsense.git
```

### 安装依赖项

```bash
sudo apt-get install -y libudev-dev pkg-config libgtk-3-dev
sudo apt-get install -y libusb-1.0-0-dev pkg-config
sudo apt-get install -y libglfw3-dev
sudo apt-get install -y libssl-dev
```

### 添加udev规则

```bash
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger 
```

### 编译librealsense

```bash
mkdir build
cd build
cmake ../ -DBUILD_EXAMPLES=true
make
sudo make install
```

### 测试安装结果

终端输入`realsense-viewer`，若能打开如下界面则安装成功。  
![Alt text](posts/2023-08-31-realsense/realsense-viewer.png)

# ROS环境开发
## 配置realsense ros

```bash
sudo apt install ros-noetic-ddynamic-reconfigure ros-noetic-rgbd-launch

cd ~/catkin_ws/src

git clone https://github.com/IntelRealSense/realsense-ros.git

cd ~/catkin_ws && catkin_make
```

**注意realsense ros1和ros2的分支**

## 测试编译结果

```bash
roslaunch realsense2_camera demo_pointcloud.launch 
```

![Alt text](posts/2023-08-31-realsense/pointcloud.png)  

**写在最后**  
使用`rosrun rqt_reconfigure rqt_reconfigure`时可能会出现

```bash
ValueError: PyCapsule_GetPointer called with incorrect name
```

的错误，解决方法如下

卸载PyQt5，替换为python3-pyqt5

```bash
sudo pip uninstall PyQt5
sudo apt-get install python3-pyqt5
```

再重新安装rqt-reconfigure

```bash
sudo apt-get purge ros-noetic-rqt-reconfigure
sudo apt-get install ros-noetic-rqt-reconfigure
```

## launch文件

- enable_depth

布尔型变量，默认为`true`，用于指定传感器是否发布深度相关Topic。其主要影响`/camera/depth/`相关Topic的发布。

```bash
/camera/depth/camera_info
/camera/depth/color/points
/camera/depth/image_rect_raw
/camera/depth/image_rect_raw/compressed
/camera/depth/image_rect_raw/compressed/parameter_descriptions
/camera/depth/image_rect_raw/compressed/parameter_updates
/camera/depth/image_rect_raw/compressedDepth
/camera/depth/image_rect_raw/compressedDepth/parameter_descriptions
/camera/depth/image_rect_raw/compressedDepth/parameter_updates
/camera/depth/image_rect_raw/theora
/camera/depth/image_rect_raw/theora/parameter_descriptions
/camera/depth/image_rect_raw/theora/parameter_updates
```

- enable_depth

布尔型变量，默认为`true`，用于指定传感器是否发布深度相关Topic。其主要影响/camera/depth/相关Topic的发布。

```bash
/camera/depth/camera_info
/camera/depth/color/points
/camera/depth/image_rect_raw
/camera/depth/image_rect_raw/compressed
/camera/depth/image_rect_raw/compressed/parameter_descriptions
/camera/depth/image_rect_raw/compressed/parameter_updates
/camera/depth/image_rect_raw/compressedDepth
/camera/depth/image_rect_raw/compressedDepth/parameter_descriptions
/camera/depth/image_rect_raw/compressedDepth/parameter_updates
/camera/depth/image_rect_raw/theora
/camera/depth/image_rect_raw/theora/parameter_descriptions
/camera/depth/image_rect_raw/theora/parameter_updates
```

- enable_infra

一共包括`enable_infra`、`enable_infra1`、`enable_infra2`三个变量，用于控制是否输出红外影像。`enable_infra1`、`enable_infra2`
分别表示左、右红外相机的影像。这三个变量默认都为false。其主要影响`/camera/infra1/`和`/camera/infra2/`相关Topic的发布。

```bash
/camera/infra1/camera_info
/camera/infra1/image_rect_raw
/camera/infra1/image_rect_raw/compressed
/camera/infra1/image_rect_raw/compressed/parameter_descriptions
/camera/infra1/image_rect_raw/compressed/parameter_updates
/camera/infra1/image_rect_raw/compressedDepth
/camera/infra1/image_rect_raw/compressedDepth/parameter_descriptions
/camera/infra1/image_rect_raw/compressedDepth/parameter_updates
/camera/infra1/image_rect_raw/theora
/camera/infra1/image_rect_raw/theora/parameter_descriptions
/camera/infra1/image_rect_raw/theora/parameter_updates

/camera/infra2/camera_info
/camera/infra2/image_rect_raw
/camera/infra2/image_rect_raw/compressed
/camera/infra2/image_rect_raw/compressed/parameter_descriptions
/camera/infra2/image_rect_raw/compressed/parameter_updates
/camera/infra2/image_rect_raw/compressedDepth
/camera/infra2/image_rect_raw/compressedDepth/parameter_descriptions
/camera/infra2/image_rect_raw/compressedDepth/parameter_updates
/camera/infra2/image_rect_raw/theora
/camera/infra2/image_rect_raw/theora/parameter_descriptions
/camera/infra2/image_rect_raw/theora/parameter_updates
```

**注意**  
`enable_infra`并不会真正影响红外影像的发布，真正影像开关的是`enable_infra1`和`enable_infra2`

- enable_color

布尔型变量，默认为`true`，用于指定RGB相机是否发布RGB影像相关Topic。其主要影响`/camera/color/`相关Topic的发布。

```bash
/camera/color/camera_info
/camera/color/image_raw
/camera/color/image_raw/compressed
/camera/color/image_raw/compressed/parameter_descriptions
/camera/color/image_raw/compressed/parameter_updates
/camera/color/image_raw/compressedDepth
/camera/color/image_raw/compressedDepth/parameter_descriptions
/camera/color/image_raw/compressedDepth/parameter_updates
/camera/color/image_raw/theora
/camera/color/image_raw/theora/parameter_descriptions
/camera/color/image_raw/theora/parameter_updates
```

- enable_gyro

布尔型变量，默认为`false`，用于指定陀螺仪是否发布相关Topic。其主要影响`/camera/gyro/`相关Topic的发布。

```bash
/camera/gyro/imu_info
/camera/gyro/sample
```

- enable_accel

布尔型变量，默认为`false`，用于指定加速度计是否发布相关Topic。其主要影响`/camera/accel/`相关Topic的发布。

```bash
/camera/accel/imu_info
/camera/accel/sample
```

- enable_sync

布尔型变量，默认为`false`，用于同步各传感器之间的数据，这样由相机发出的各个Topic的频率就能相同。

- align_depth

布尔型变量，默认为`false`
，用于将图像对齐，其并不会修改之前发出的Topic，而是重新发出两类多个Topic：`/camera/aligned_depth_to_color`
和`/camera/aligned_depth_to_infra1`。

```bash
/camera/aligned_depth_to_color/camera_info
/camera/aligned_depth_to_color/image_raw
/camera/aligned_depth_to_color/image_raw/compressed
/camera/aligned_depth_to_color/image_raw/compressed/parameter_descriptions
/camera/aligned_depth_to_color/image_raw/compressed/parameter_updates
/camera/aligned_depth_to_color/image_raw/compressedDepth
/camera/aligned_depth_to_color/image_raw/compressedDepth/parameter_descriptions
/camera/aligned_depth_to_color/image_raw/compressedDepth/parameter_updates
/camera/aligned_depth_to_color/image_raw/theora
/camera/aligned_depth_to_color/image_raw/theora/parameter_descriptions
/camera/aligned_depth_to_color/image_raw/theora/parameter_updates

/camera/aligned_depth_to_infra1/camera_info
/camera/aligned_depth_to_infra1/image_raw
/camera/aligned_depth_to_infra1/image_raw/compressed
/camera/aligned_depth_to_infra1/image_raw/compressed/parameter_descriptions
/camera/aligned_depth_to_infra1/image_raw/compressed/parameter_updates
/camera/aligned_depth_to_infra1/image_raw/compressedDepth
/camera/aligned_depth_to_infra1/image_raw/compressedDepth/parameter_descriptions
/camera/aligned_depth_to_infra1/image_raw/compressedDepth/parameter_updates
/camera/aligned_depth_to_infra1/image_raw/theora
/camera/aligned_depth_to_infra1/image_raw/theora/parameter_descriptions
/camera/aligned_depth_to_infra1/image_raw/theora/parameter_updates
```

## 启动相机节点

```bash
roslaunch realsense2_camera rs_camera.launch
```

### 发布的Topic

- 带有info的Topic: 获得传感器相关信息
- 带有compress的Topic: 经过压缩后的数据流，大小更小。但同时数据的质量可能会有一定的下降
- 带有raw的Topic: 与压缩的数据流对应，未经过压缩的原始数据流
- 带有theora的Topic: 与不带theora的Topic相同，差别在于编码方式
- 带有rect的Topic: 校正后的数据，Rectify的缩写
- 其它Topic: 获得加速度计或陀螺仪的数据，订阅/camera/gyro/sample和/camera/accel/sample

### 修改参数
#### 支持的分辨率和帧率

- 单目相机(RGB影像)
  - 分辨率： 320×180,320×240,424×240,640×360,640×480,848×480,960×540,1280×720,1920×1080
  - 帧率： 6,15,30,60
- 双目红外相机(灰度影像&深度影像)
  - 分辨率： 256×144,424×240,480×270,640×360,640×400,640×480,848×100,848×480,1280×720,1280×800
  - 帧率： 6,15,25,30,60,90,100,300
- IMU(加速度计&陀螺仪)
  - 加速度计帧率： 63,250
  - 陀螺仪帧率： 200,400

> 注：需要同时修改分辨率和帧率才能保证参数生效

### camera_info参数

以彩色图像为例

```bash
header: 
  seq: 1
  stamp: 
    secs: 1693631864
    nsecs:  91671944
  frame_id: "camera_color_optical_frame"
height: 480
width: 640
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [618.30322265625, 0.0, 326.67083740234375, 0.0, 618.4461669921875, 244.36752319335938, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [618.30322265625, 0.0, 326.67083740234375, 0.0, 0.0, 618.4461669921875, 244.36752319335938, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---
```

- header: 标准消息头  
  seq: 表示消息的序列ID，连续递增  
  stamp: 包括两个字段 secs（秒）和nsecs（纳秒），表示消息的时间戳  
  frame_id：与此数据相关联的帧ID
- height: 图像的高度，以像素为单位
- width: 图像的宽度，以像素为单位
- distortion_model: 相机的畸变模型，通常是"plumb_bob"，表示针孔相机模型
- D: 畸变系数数组，取决于畸变模型，通常包含5个畸变系数，用于校正畸变 $(k1, k2, t1, t2, k3)$
- K: 相机内参矩阵，包含9个元素，其中 fx 和 fy 是焦距，cx 和 cy 是光学中心的坐标，内参矩阵可以将相机坐标中的3D点投影到2D像素坐标

$$
k = \begin{bmatrix}
fx & 0 & cx \\
0 & fy & cy \\
0 & 0 & 1
\end{bmatrix}
$$

- R: 旋转矩阵，通常是单位矩阵，表示相机坐标系和图像坐标系之间的旋转关系
- P: 投影矩阵，包含12个元素，用于将相机坐标系中的点映射到图像平面上

$$
p = \begin{bmatrix}
fx^{'} & 0 & cx^{'} & Tx \\
0 & fy^{'} & cy^{'} & Ty \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

左侧3*3矩阵是相机的内参矩阵，可能与相机内参K不同。对于单目相机Tx=Ty=0。对于双目相机，Tx和Ty有所不同。

- binning_x: 沿x轴的像素采样因子，通常为0
- binning_y: 沿y轴的像素采样因子，通常为0
- roi: 感兴趣区域（ROI）的描述，包括x和y偏移、高度、宽度以及是否进行校正
