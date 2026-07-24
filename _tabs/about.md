---
icon: fas fa-user
order: 4
---

<div class="about-hero">
  <span class="about-kicker">ABOUT · NOCTILUMEE</span>
  <h2>NOCTILUMEE</h2>
  <p>这个博客主要用来放自己的学习笔记、踩坑记录和菜谱。</p>
  <p>
    平时写得比较多的是计算机视觉、机器人、C++ / Python、Linux 和各类开发工具。
    文章不一定都很完整，但我会尽量把实际用过的命令、配置和思路记清楚，方便以后查，也希望能帮到遇到同类问题的人。
  </p>
  <div class="about-actions">
    <a class="about-link" href="https://github.com/du-xinyi" target="_blank" rel="noopener noreferrer">
      <i class="fab fa-github"></i>&nbsp;GitHub
    </a>
    <a class="about-link" href="https://twitter.com/noctilumee" target="_blank" rel="noopener noreferrer">
      <i class="fab fa-twitter"></i>&nbsp;Twitter
    </a>
    <a class="about-link" href="mailto:du-xinyi@foxmail.com">
      <i class="fas fa-envelope"></i>&nbsp;Email
    </a>
  </div>
</div>

<div class="about-grid">
  <section class="about-card">
    <div class="about-card-icon"><i class="fas fa-code"></i></div>
    <h3>技术笔记</h3>
    <p>C++、Python、Linux、Docker、Git，还有一些环境配置和调试记录。</p>
  </section>

  <section class="about-card">
    <div class="about-card-icon"><i class="fas fa-robot"></i></div>
    <h3>视觉与机器人</h3>
    <p>图像处理、坐标变换、深度学习、机器人相关的内容，主要偏学习笔记。</p>
  </section>

  <section class="about-card">
    <div class="about-card-icon"><i class="fas fa-terminal"></i></div>
    <h3>工具折腾</h3>
    <p>编辑器、终端、脚本和部署流程。怎么省事怎么来，踩过的坑也顺手记下来。</p>
  </section>

  <section class="about-card">
    <div class="about-card-icon"><i class="fas fa-utensils"></i></div>
    <h3>生活记录</h3>
    <p>以菜谱为主，偶尔也记一点别的生活片段。</p>
  </section>
</div>

<div class="about-card about-wide">
  <h3>常写的内容</h3>
  <div class="about-tags">
    <span>C++</span>
    <span>Python</span>
    <span>Linux</span>
    <span>OpenCV</span>
    <span>PyTorch</span>
    <span>ROS</span>
    <span>Docker</span>
    <span>Git</span>
    <span>Debugging</span>
    <span>Embedded</span>
    <span>Cooking</span>
  </div>
</div>

<div class="about-card about-wide">
  <h3>最近更新</h3>
  <ul class="about-recent-list">
    {% for post in site.posts limit: 3 %}
      <li>
        <a class="about-post-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: '%Y-%m-%d' }}</time>
      </li>
    {% endfor %}
  </ul>
</div>
