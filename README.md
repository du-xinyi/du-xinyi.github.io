# 守心如一

基于 Jekyll 与 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 主题搭建的个人博客，记录技术学习、工程实践与生活里的菜谱与片段。

| 入口 | 地址 |
| --- | --- |
| 站点 | <https://du-xinyi.github.io/> |
| 归档 | <https://du-xinyi.github.io/archives/> |
| 关于 | <https://du-xinyi.github.io/about/> |

## 内容方向

- **技术笔记**：C++、Python、Linux、Docker、Git 等工具与技术栈。
- **工程实践**：环境配置、调试流程、部署经验与开发效率工具。
- **学习记录**：图像处理、计算机视觉、机器人学、深度学习等方向。
- **生活菜谱**：反复验证过的家常菜、甜品与饮品配方。

## 技术栈

- Jekyll `~> 4.3` 与 Chirpy `7.6.0`
- Ruby `~> 3.1`，CI 使用 Ruby `3.4`
- Node.js `24` 与 npm
- Bootstrap `5`、SCSS、Rollup、PurgeCSS
- Giscus 评论、PWA 离线缓存、GitHub Pages 部署

## 环境要求

- Ruby `3.1+`，推荐使用与 CI 一致的 Ruby `3.4`
- Bundler
- Node.js `24` 与 npm

## 本地运行

克隆仓库并安装依赖：

```bash
git clone https://github.com/du-xinyi/du-xinyi.github.io.git
cd du-xinyi.github.io
bundle install
npm ci --ignore-scripts --no-audit --no-fund
```

构建前端资源并启动开发服务器：

```bash
npm run build
bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

打开 <http://127.0.0.1:4000/> 预览站点。

也可以使用仓库提供的脚本，默认监听 `127.0.0.1:3000`：

```bash
bash tools/run.sh
```

以线上模式构建并预览：

```bash
npm run build
JEKYLL_ENV=production bundle exec jekyll serve --host 127.0.0.1 --port 4000
```

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `npm run build` | 同时构建 CSS 与 JavaScript |
| `npm run build:css` | 通过 PurgeCSS 生成生产环境 Bootstrap 样式 |
| `npm run build:js` | 通过 Rollup 构建生产环境 JavaScript |
| `npm run watch:js` | 监听并持续构建 JavaScript |
| `npm run lint:js` | 运行 ESLint |
| `npm run lint:scss` | 运行 Stylelint |
| `npm test` | 运行 ESLint 与 Stylelint |
| `bundle exec jekyll clean` | 清理 Jekyll 构建缓存 |
| `bash tools/test.sh` | 构建生产站点并运行 html-proofer |

## 目录结构

```text
_posts/             博客文章
_tabs/              首页之外的固定页面，如 About、Archives
_layouts/           页面布局
_includes/          可复用的页面片段
_sass/custom/       自定义玻璃质感与页面样式
_sass/vendors/      PurgeCSS 生成的第三方样式，不手动编辑
_javascript/        JavaScript 源码
assets/js/dist/     Rollup 生成的 JavaScript
assets/img/         头像、文章图片与站点图标
_data/              导航、联系方式与本地化文案
tools/              本地运行与站点测试脚本
.github/workflows/  GitHub Pages 自动部署流程
```

## 写文章

在 `_posts/` 中新建 `YYYY-MM-DD-title.md`，并添加 front matter：

```yaml
---
title: Linux Tools
description: 整理 Linux 下 SSH、压缩、传输、检索及系统管理等常用工具
date: 2025-10-31 12:30:00 +0800
categories: [ Systems, Linux ]
tags: [ Linux, CLI, SSH ]
---
```

常用可选字段：

- `pin: true`：将文章置顶。
- `math: true`：启用数学公式渲染。
- `image:`：为首页卡片和社交分享配置封面图。

## 样式说明

自定义样式集中放在 `_sass/custom/`。生产环境使用的 `_sass/vendors/_bootstrap.scss` 由 `npm run build:css` 生成，不直接手动修改；CI 会在部署前重新执行完整前端构建，避免线上使用过期样式。

## 部署

推送到 `master`（或 `main`）后，`.github/workflows/pages-deploy.yml` 会依次完成依赖安装、前端构建、lint、Jekyll 生产构建、html-proofer 检查与 GitHub Pages 部署。

站点启用了 PWA 缓存。若更新后浏览器仍显示旧页面，点击站点右上角的刷新按钮，或在浏览器中强制刷新。

## 许可证

站点代码遵循 [MIT License](LICENSE)。文章内容按站点页脚声明采用 CC BY 4.0。

## 致谢

- [Jekyll](https://jekyllrb.com/)
- [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)
