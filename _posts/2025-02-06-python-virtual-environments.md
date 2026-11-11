---
title: Python Virtual Environments
description: 整理 Python 虚拟环境的用途，以及 venv、Conda、uv 三种工具的环境创建、包管理与项目依赖管理方法
date: 2025-02-06 12:30:00 +0800
categories: [ Development, Python ]
tags: [ Python, venv, Conda, uv ]
---

不同的项目往往依赖同一个包的不同版本，全部装进系统的 Python 会互相冲突。虚拟环境为每个项目提供一套独立的解释器和包目录，互不影响，删除时也只需要删掉对应的目录

Ubuntu 23.04 起，系统自带的`pip`直接安装包时会报`externally-managed-environment`错误，这是为了防止破坏系统自身依赖的 Python 包，正确的做法就是在虚拟环境中安装

## 工具对比

| 工具    | 管理 Python 版本            | 包的来源                  | 特点                                                |
| :------ | :-------------------------- | :------------------------ | :-------------------------------------------------- |
| `venv`  | 不能，使用系统已有的 Python | PyPI                      | 标准库自带，不需要额外安装，功能最基础              |
| `conda` | 能                          | Conda 频道，也可以用`pip` | 可以安装 CUDA、编译器等非 Python 依赖，环境集中存放 |
| `uv`    | 能                          | PyPI                      | 速度快，带锁文件的项目依赖管理，兼容`pip`的用法     |

## venv

`venv`是 Python 标准库自带的虚拟环境模块，环境就是项目目录下的一个文件夹，习惯上命名为`.venv`

```bash
sudo apt install -y python3-venv
```
{: file='Ubuntu 需要先安装 venv 模块'}

```bash
python3 -m venv .venv
```
{: file='在当前目录创建虚拟环境'}

```bash
source .venv/bin/activate
deactivate
```
{: file='激活与退出环境'}

激活后终端提示符前会出现`(.venv)`，此时的`python`和`pip`都指向环境内部，安装的包只存在于该环境中

```bash
pip install numpy
pip list
```
{: file='安装包、查看已安装的包'}

```bash
pip freeze > requirements.txt
```
{: file='导出当前环境的包及版本'}

```bash
pip install -r requirements.txt
```
{: file='根据 requirements.txt 还原环境'}

```bash
rm -rf .venv
```
{: file='删除环境'}

虚拟环境中保存的是绝对路径，移动或重命名项目目录后环境会失效，需要删除后重新创建。`.venv`不应提交到代码仓库，提交`requirements.txt`即可

## Conda

`conda`是`anaconda`发行版中的包管理器，用于安装、更新、卸载软件包，以及创建和管理不同的`Python`环境。除`Anaconda`外，也可以使用默认`conda-forge`频道的`Miniforge`发行版（`Anaconda`的`defaults`频道对组织用户有商业授权限制）

### 环境管理

```bash
conda create --name env
```
{: file='创建 env 环境'}

```bash
conda create --name env python=3.12
```
{: file='创建 Python 版本为 3.12 的 env 环境'}

```bash
conda env create -f conda_environment.yaml
```
{: file='使用 conda_environment 创建环境'}

```bash
conda env list
```
{: file='查看已创建的环境'}

```bash
conda activate env
conda deactivate
```
{: file='激活与退出环境'}

```bash
conda create --name env_copy --clone env
```
{: file='克隆环境'}

```bash
conda env export --name env > conda_environment.yaml
conda env export --name env --from-history > conda_environment.yaml
```
{: file='导出环境'}

直接导出会包含所有依赖及其构建编号，只能在相同的操作系统上还原；加上`--from-history`只导出手动安装过的包，适合跨平台共享

```bash
conda env remove --name env
```
{: file='删除环境'}

```bash
conda config --set auto_activate_base false
```
{: file='打开终端时不自动激活 base 环境'}

### 包管理

以下命令作用于当前激活的环境，也可以用`--name env`指定环境

```bash
conda install numpy
conda install numpy=1.26
conda install -c conda-forge opencv
```
{: file='安装包、指定版本、指定频道'}

```bash
conda list
conda search numpy
```
{: file='查看已安装的包、搜索可用版本'}

```bash
conda update numpy
conda remove numpy
```
{: file='更新与卸载'}

```bash
conda clean --all
```
{: file='清理下载缓存和未使用的包'}

`conda`环境中可以使用`pip`安装`conda`频道中没有的包，但应先用`conda`安装能安装的包，最后再使用`pip`，否则`conda`之后的依赖解析可能会破坏`pip`安装的包

### 优化指令

#### 显示配置的来源文件及其内容

```bash
conda config --show-sources
```

与`conda config --show channels`相比，能查看当前环境所读取的配置文件信息

#### 使用 libmamba 求解器

conda 23.10 起默认使用`libmamba`求解器（即`mamba`的求解器），无需另外安装`mamba`即可获得更快的依赖解析。可通过以下命令确认

```bash
conda config --show solver
```

如需使用`mamba`命令本身，官方推荐直接安装`Miniforge`发行版（自带`conda-forge`与`mamba`），不建议在 Anaconda 的`base`环境中用`conda install mamba`混装`conda-forge`的包。`mamba`的大部分命令与`conda`相同，但 2.x 版本在部分子命令上存在差异

## uv

`uv`是用 Rust 编写的 Python 包与项目管理工具，把`pip`、`venv`、Python 版本管理和依赖锁定合并到了一个命令里，安装包的速度比`pip`快很多

### 安装

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
{: file='安装 uv'}

安装到`~/.local/bin`，重新打开终端后生效

```bash
uv self update
```
{: file='更新 uv'}

### 管理 Python 版本

`uv`可以自行下载并管理多个 Python 版本，不依赖系统自带的 Python，也不需要`sudo`

```bash
uv python list
```
{: file='查看已安装和可下载的版本'}

```bash
uv python install 3.12
```
{: file='安装指定版本'}

```bash
uv python pin 3.12
```
{: file='指定当前项目使用的版本，写入 .python-version'}

创建环境或运行项目时，所需的 Python 版本不存在会自动下载

### 虚拟环境与 pip 接口

`uv pip`的用法与`pip`基本一致，适合在已有的`requirements.txt`工作流中直接替换

```bash
uv venv
uv venv --python 3.12
```
{: file='创建 .venv，或指定 Python 版本创建'}

```bash
source .venv/bin/activate
```
{: file='激活环境，与 venv 相同'}

```bash
uv pip install numpy
uv pip install -r requirements.txt
```
{: file='安装包'}

```bash
uv pip list
uv pip freeze > requirements.txt
```
{: file='查看与导出'}

`uv pip`默认操作当前目录下的`.venv`，不激活环境也可以使用。`uv venv`创建的环境中没有`pip`命令，安装包一律使用`uv pip`

### 项目管理

`uv`推荐的用法是把依赖写进`pyproject.toml`，由它负责创建环境、解析依赖和生成锁文件，不需要手动激活环境

```bash
uv init --app --no-package demo
cd demo
```
{: file='新建项目'}

`--app --no-package`创建的是只有一个`main.py`的脚本项目；不加参数时会生成带`src/`目录、可以被安装的包

| 文件              | 作用                                                   |
| :---------------- | :----------------------------------------------------- |
| `pyproject.toml`  | 项目信息和依赖列表                                     |
| `uv.lock`         | 锁文件，记录每个依赖解析出的确切版本，应提交到代码仓库 |
| `.python-version` | 项目使用的 Python 版本                                 |
| `.venv`           | 虚拟环境，首次运行时自动创建，不提交到代码仓库         |

```bash
uv add numpy
uv add "numpy>=2.0"
uv add --dev pytest
```
{: file='添加依赖、指定版本范围、添加开发依赖'}

```bash
uv remove numpy
```
{: file='移除依赖'}

```bash
uv run main.py
uv run pytest
```
{: file='在项目环境中运行脚本或命令'}

`uv run`运行前会检查环境与锁文件是否一致，不一致时自动同步，因此不需要激活环境

```bash
uv sync
```
{: file='按锁文件创建或更新环境，克隆项目后执行'}

```bash
uv lock --upgrade
```
{: file='把依赖升级到允许范围内的最新版本'}

```bash
uv tree
```
{: file='查看依赖树'}

```bash
uv export --format requirements-txt -o requirements.txt
```
{: file='导出为 requirements.txt'}

### 运行命令行工具

`ruff`、`black`这类命令行工具不属于某个项目，可以用`uvx`在临时环境中直接运行，不污染任何环境

```bash
uvx ruff check
```
{: file='临时运行，等价于 uv tool run ruff check'}

```bash
uv tool install ruff
uv tool list
```
{: file='安装为全局可用的命令'}

### 镜像与缓存

下载速度慢时可以配置 PyPI 镜像，对所有项目生效

```toml
[[index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```
{: file='~/.config/uv/uv.toml'}

只想对单个项目生效时，在`pyproject.toml`中把`[[index]]`写成`[[tool.uv.index]]`

`uv`下载过的包保存在全局缓存中，多个环境通过链接共用同一份文件，不会重复占用磁盘

```bash
uv cache dir
uv cache clean
```
{: file='查看缓存位置、清理缓存'}

## 如何选择

- 只是临时装几个包、不想装额外的工具，使用`venv`
- 需要 CUDA、OpenCV 等带有非 Python 依赖的包，或者要与他人共享完整的运行环境，使用`conda`
- 新建的纯 Python 项目使用`uv`，依赖有锁文件，换一台机器执行`uv sync`就能得到相同的环境
- 不要在同一个环境里混用多种工具安装包，`conda`环境中确实需要`pip`时放在最后安装
