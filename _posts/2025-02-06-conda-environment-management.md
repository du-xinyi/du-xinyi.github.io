---
title: Conda Environment Management
description: 整理 Conda 环境创建、包管理、配置查看与常用命令
date: 2025-02-06 12:30:00 +0800
categories: [ Development, Python ]
tags: [ Conda, Python ]
---

# conda相关笔记

`conda`是`anaconda`发行版中的包管理器，用于安装、更新、卸载软件包，以及创建和管理不同的`Python`环境。除`Anaconda`外，也可以使用默认`conda-forge`频道的`Miniforge`发行版（`Anaconda`的`defaults`频道对组织用户有商业授权限制）

## 环境管理

```bash
conda create --name env
```
{: file='创建env环境'}

```bash
conda create --name env python=3.8
```
{: file='创建python版本为3.8的env环境'}

```bash
conda env create -f conda_environment.yaml
```
{: file='使用conda_environment创建环境'}

```bash
conda env list
```
{: file='查看已创建的环境'}

## 优化指令
### 显示配置的来源文件及其内容

```bash
conda config --show-sources
```

与`conda config --show channels`相比，能查看当前环境所读取的配置文件信息

### 使用libmamba求解器

conda 23.10 起默认使用`libmamba`求解器（即`mamba`的求解器），无需另外安装`mamba`即可获得更快的依赖解析。可通过以下命令确认

```bash
conda config --show solver
```

如需使用`mamba`命令本身，官方推荐直接安装`Miniforge`发行版（自带`conda-forge`与`mamba`），不建议在Anaconda的`base`环境中用`conda install mamba`混装`conda-forge`的包。`mamba`的大部分命令与`conda`相同，但2.x版本在部分子命令上存在差异
