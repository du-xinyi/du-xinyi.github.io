---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

<div class="about-hero">
  <span class="about-kicker">ABOUT ME</span>
  <h2>你好，我是 du-xinyi</h2>
  <p>
    这里记录我在软件开发、Linux 工具链、嵌入式与机器人、图像处理和深度学习等方向的学习笔记，
    也收录一些菜谱和生活片段。比起追求完美成文，我更在意把问题、过程和可复用的结论清楚地留下来。
  </p>
  <div class="about-actions">
    <a class="about-link" href="https://github.com/du-xinyi" target="_blank" rel="noopener noreferrer">GitHub</a>
    <a class="about-link" href="mailto:du-xinyi@foxmail.com">Email</a>
    <a class="about-link" href="https://twitter.com/noctilumee" target="_blank" rel="noopener noreferrer">Twitter</a>
  </div>
</div>

<div class="about-grid">
  <section class="about-card">
    <span class="about-card-icon"><i class="fas fa-code"></i></span>
    <h3>技术笔记</h3>
    <p>整理 C++、Python、Linux、Docker、Git 等工具与技术栈的使用方法。</p>
  </section>

  <section class="about-card">
    <span class="about-card-icon"><i class="fas fa-screwdriver-wrench"></i></span>
    <h3>工程实践</h3>
    <p>记录环境配置、调试流程、部署经验和开发效率工具的实践心得。</p>
  </section>

  <section class="about-card">
    <span class="about-card-icon"><i class="fas fa-brain"></i></span>
    <h3>学习记录</h3>
    <p>覆盖图像处理、机器人学、深度学习等方向的基础概念与实现细节。</p>
  </section>

  <section class="about-card">
    <span class="about-card-icon"><i class="fas fa-utensils"></i></span>
    <h3>生活菜谱</h3>
    <p>保存反复验证过的家常菜、甜品和饮品配方，方便随时查阅。</p>
  </section>
</div>

<div class="about-card about-wide">
  <h3>长期关注</h3>
  <div class="about-tags">
    <span>C++</span>
    <span>Python</span>
    <span>Linux</span>
    <span>Docker</span>
    <span>Git</span>
    <span>OpenCV</span>
    <span>PyTorch</span>
    <span>ROS</span>
    <span>Embedded</span>
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
