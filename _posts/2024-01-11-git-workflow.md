---
title: Git Workflow
description: 整理 Git 的基本配置、日常提交、忽略文件、分支与标签、撤销修改、同步上游仓库与子模块的用法
date: 2024-01-11 12:30:00 +0800
categories: [ Development, Tooling ]
tags: [ Git, Version Control, GitHub ]
---

## 基本配置

```bash
git config --global user.name "name"
git config --global user.email "email@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase true
```
{: file='用户信息与默认行为'}

`pull.rebase`设为`true`后，`git pull`会把本地提交变基到远程提交之后，避免产生多余的合并提交

```bash
git config --global --list
```
{: file='查看全局配置'}

## 日常提交

```bash
git status
git diff
```
{: file='查看工作区状态与未暂存的修改'}

```bash
git add <文件>
git add -p
```
{: file='暂存整个文件，或逐块选择要暂存的修改'}

```bash
git commit -m "提交说明"
git push
```
{: file='提交并推送'}

```bash
git log --oneline --graph --all
```
{: file='以图形方式查看所有分支的提交历史'}

## 忽略文件

不需要纳入版本管理的文件（编译产物、依赖目录、IDE 配置、密钥等）写入仓库根目录的`.gitignore`

```text
# 忽略所有 .log 文件
*.log

# 忽略任意位置的 build 目录
build/

# 只忽略根目录下的 config.json
/config.json

# 不忽略 important.log
!important.log
```
{: file='.gitignore'}

`.gitignore`只对尚未被跟踪的文件生效。文件已经提交过时，需要先从版本库中移除（保留本地文件）

```bash
git rm --cached <文件>
git rm -r --cached <目录>
```
{: file='停止跟踪已提交的文件'}

```bash
git check-ignore -v <文件>
```
{: file='查看文件被哪条规则忽略'}

## 分支操作

```bash
git switch -c feature
```
{: file='创建并切换到新分支'}

```bash
git switch main
git merge feature
```
{: file='把 feature 分支合并到 main'}

```bash
git switch feature
git rebase main
```
{: file='把 feature 分支的提交变基到 main 的最新提交之后'}

变基会改写提交历史，只应对尚未推送或仅自己使用的分支进行

![合并前、merge 之后与 rebase 之后的提交历史对比](posts/2024-01-11-git-workflow/merge_rebase.png)

```bash
git push -u origin feature
```
{: file='首次推送分支并建立跟踪关系'}

```bash
git branch -d feature
git push origin --delete feature
```
{: file='删除本地分支与远程分支'}

### 挑选提交

```bash
git cherry-pick <提交>
```
{: file='把其他分支上的某个提交应用到当前分支'}

遇到冲突时，解决冲突并`git add`后执行`git cherry-pick --continue`，放弃则执行`git cherry-pick --abort`

## 标签

标签用于标记某个提交，通常用来标记发布版本

```bash
git tag v1.0.0
git tag -a v1.0.0 -m "版本说明"
```
{: file='创建轻量标签，或带说明的附注标签'}

```bash
git tag
git show v1.0.0
```
{: file='查看所有标签、查看标签详情'}

```bash
git push origin v1.0.0
git push origin --tags
```
{: file='推送单个标签、推送所有标签'}

`git push`默认不会推送标签，需要单独推送

```bash
git tag -d v1.0.0
git push origin --delete v1.0.0
```
{: file='删除本地标签与远程标签'}

## 撤销修改

```bash
git restore <文件>
```
{: file='丢弃工作区中未暂存的修改（不可恢复）'}

```bash
git restore --staged <文件>
```
{: file='取消暂存，保留工作区的修改'}

```bash
git commit --amend
```
{: file='修改最近一次提交的内容或说明'}

```bash
git reset --soft HEAD~1
```
{: file='撤销最近一次提交，修改保留在暂存区'}

```bash
git reset --hard HEAD~1
```
{: file='撤销最近一次提交，并丢弃其修改（不可恢复）'}

```bash
git revert <提交>
```
{: file='新建一个提交来抵消指定提交，适用于已推送的提交'}

`--amend`、`reset`会改写历史，已经推送到公共分支的提交应使用`revert`撤销。误操作后可以用`git reflog`找回被`reset`掉的提交

### 暂存现场

```bash
git stash
git stash list
git stash pop
```
{: file='临时保存未提交的修改，之后再恢复'}

## 同步上游仓库

fork 是在自己账号下复制出的一个独立仓库，不会自动跟随源仓库更新。以本博客为例，源仓库（上游）为[jekyll-theme-chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)，fork 得到的仓库为[du-xinyi.github.io](https://github.com/du-xinyi/du-xinyi.github.io)

![上游仓库、fork 得到的仓库与本地仓库之间的关系](posts/2024-01-11-git-workflow/fork_upstream.png)

```bash
git clone https://github.com/du-xinyi/du-xinyi.github.io.git
```
{: file='克隆 fork 得到的仓库'}

此时远程仓库只有`origin`，指向自己的仓库。把源仓库添加为另一个远程仓库，习惯上命名为`upstream`

```bash
git remote add upstream https://github.com/cotes2020/jekyll-theme-chirpy.git
git remote -v
```
{: file='添加上游仓库'}

```text
origin    https://github.com/du-xinyi/du-xinyi.github.io.git (fetch)
origin    https://github.com/du-xinyi/du-xinyi.github.io.git (push)
upstream  https://github.com/cotes2020/jekyll-theme-chirpy.git (fetch)
upstream  https://github.com/cotes2020/jekyll-theme-chirpy.git (push)
```
{: file='远程仓库列表'}

```bash
git fetch upstream
git merge upstream/master
```
{: file='拉取上游的更新并合并到当前分支'}

有冲突时，手动修改冲突文件后执行`git add`和`git commit`完成合并，最后`git push`推送到自己的仓库

## 子模块

仓库通过`submodule`引用了其他仓库时，直接克隆得到的子模块目录是空的，需要在克隆时一并拉取

```bash
git clone --recurse-submodules <仓库地址>
```
{: file='克隆仓库及其子模块'}

```bash
git submodule update --init --recursive
```
{: file='为已经克隆的仓库补充拉取子模块'}

```bash
git submodule update --remote
```
{: file='把子模块更新到其远程分支的最新提交'}
