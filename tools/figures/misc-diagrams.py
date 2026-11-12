import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import *
Z = '2025-05-17-zeromq-messaging'

# ---- A. patterns
fig, ax = canvas(15, 4.6, 0.8)
# REQ/REP
label(ax, 2.3, 4.25, '请求-响应 REQ/REP', 12, bold=True)
box(ax, 0.3, 1.9, 1.5, 1.0, '客户端\nREQ', BLUE); box(ax, 2.9, 1.9, 1.5, 1.0, '服务端\nREP', GREEN)
arrow(ax, (1.8, 2.65), (2.9, 2.65), '① 请求', off=(0, 0.2)); arrow(ax, (2.9, 2.15), (1.8, 2.15), '② 响应', off=(0, -0.22))
label(ax, 2.35, 1.1, '严格按一发一收交替', 9.5, '#868e96')
# PUB/SUB
label(ax, 7.5, 4.25, '发布-订阅 PUB/SUB', 12, bold=True)
box(ax, 5.3, 1.9, 1.5, 1.0, '发布端\nPUB', ORANGE)
for i, (y, t) in enumerate([(3.1, '订阅 "status"'), (1.9, '订阅 "log"'), (0.7, '订阅 ""（全部）')]):
    box(ax, 8.3, y, 1.9, 0.8, 'SUB\n' + t, BLUE, fs=9)
    arrow(ax, (6.8, 2.4), (8.3, y + 0.4), color='#d9822b')
label(ax, 6.05, 1.1, '单向广播，按前缀过滤', 9.5, '#868e96')
# PUSH/PULL
label(ax, 12.7, 4.25, '任务分发 PUSH/PULL', 12, bold=True)
box(ax, 10.9, 1.9, 1.5, 1.0, '分发端\nPUSH', PURPLE)
for i, y in enumerate([3.1, 1.9, 0.7]):
    box(ax, 13.3, y, 1.4, 0.8, f'PULL\n工作进程{i + 1}', GREEN, fs=9)
    arrow(ax, (12.4, 2.4), (13.3, y + 0.4), f'任务{i + 1}', color='#7a4fd0', off=[(-0.3, 0.2), (0.0, 0.2), (-0.3, -0.2)][i], fs=8.5)
label(ax, 11.65, 1.1, '每条消息只给一个接收端', 9.5, '#868e96')
for x in (4.85, 10.5): ax.plot([x, x], [0.4, 4.4], color='#dee2e6', lw=1)
save(fig, Z, 'patterns.png')

# ---- B. broker
fig, ax = canvas(12, 4.2, 0.85)
for i, y in enumerate([3.0, 1.7, 0.4]):
    box(ax, 0.3, y, 1.7, 0.8, f'客户端{i + 1}\nREQ', BLUE, fs=9.5)
    arrow(ax, (2.0, y + 0.4), (4.2, 2.1), rad=0.0)
    box(ax, 10.0, y, 1.7, 0.8, f'服务端{i + 1}\nREP', GREEN, fs=9.5)
    arrow(ax, (7.8, 2.1), (10.0, y + 0.4), rad=0.0)
box(ax, 4.2, 0.9, 3.6, 2.4, '', GREY, round=0.12)
label(ax, 6.0, 3.0, '代理 broker.py', 10.5, bold=True)
box(ax, 4.4, 1.5, 1.4, 1.1, 'ROUTER\n:5559', ORANGE, fs=10); box(ax, 6.2, 1.5, 1.4, 1.1, 'DEALER\n:5560', ORANGE, fs=10)
arrow(ax, (5.8, 2.05), (6.2, 2.05), style='<|-|>')
label(ax, 6.0, 1.15, 'zmq.proxy 双向转发', 8.5, '#868e96')
label(ax, 3.0, 3.75, '记录请求来源\n回复时送回对应客户端', 8.5, '#868e96'); label(ax, 9.0, 3.75, '把请求轮询分配\n给各个服务端', 8.5, '#868e96')
save(fig, Z, 'broker.png')

# ---- C. curve
fig, ax = canvas(13, 5.6, 0.85)
box(ax, 0.3, 2.6, 3.3, 2.6, '', BLUE, round=0.12); label(ax, 1.95, 4.85, '客户端', 11.5, bold=True)
label(ax, 1.95, 3.7, '持有：\n客户端公钥 + 私钥\n服务端公钥', 9.5)
box(ax, 6.2, 2.6, 3.3, 2.6, '', GREEN, round=0.12); label(ax, 7.85, 4.85, '服务端', 11.5, bold=True)
label(ax, 7.85, 3.7, '持有：\n服务端公钥 + 私钥\nCURVE_SERVER = 1', 9.5)
arrow(ax, (3.6, 4.5), (6.2, 4.5), '① 发起握手（用服务端公钥加密）', off=(0, 0.22), fs=8.8)
arrow(ax, (6.2, 3.9), (3.6, 3.9), '② 证明持有服务端私钥', off=(0, 0.2), fs=8.8)
arrow(ax, (3.6, 3.3), (6.2, 3.3), '③ 出示客户端公钥', off=(0, 0.2), fs=8.8)
arrow(ax, (3.6, 2.85), (6.2, 2.85), '⑥ 认证通过后加密传输数据', off=(0, -0.22), fs=8.8, style='<|-|>', color='#2e9950')
box(ax, 9.9, 0.3, 2.9, 1.9, '', ORANGE, round=0.12); label(ax, 11.35, 1.85, '认证线程（ZAP）', 10.5, bold=True)
label(ax, 11.35, 1.3, 'inproc://zeromq.zap.01', 8.2, mono=True); label(ax, 11.35, 0.75, '查授权公钥列表', 9.2)
arrow(ax, (9.2, 2.6), (10.6, 2.2), '④ 客户端公钥', off=(1.0, 0.25), fs=8.8, rad=-0.15)
arrow(ax, (9.9, 1.0), (8.2, 2.6), '⑤ 200 放行 / 400 拒绝', off=(-1.45, -0.25), fs=8.8, rad=-0.2)
box(ax, 0.3, 0.3, 5.9, 1.5, '', GREY, round=0.12)
label(ax, 3.25, 1.05, '客户端通过服务端公钥确认服务端身份（①②）\n服务端通过授权列表确认客户端身份（③④⑤）', 9.3)
save(fig, Z, 'curve.png')

# ---- D. git merge / rebase
G = '2024-01-11-git-workflow'
def commit(ax, x, y, t, c):
    ax.add_patch(Circle((x, y), 0.3, fc=c[0], ec=c[1], lw=1.6, zorder=3)); label(ax, x, y, t, 9.5, bold=True)
def line(ax, a, b): ax.plot([a[0], b[0]], [a[1], b[1]], color='#868e96', lw=1.6, zorder=1)
fig, ax = canvas(16.4, 4.2, 0.8)
titles = ['合并前', 'git merge feature（在 main 上执行）', 'git rebase main（在 feature 上执行）']
for k, x0 in enumerate([0.3, 5.2, 10.7]):
    label(ax, x0 + 2.2, 3.85, titles[k], 11, bold=True)
    m = [(x0 + 0.4 + i * 0.95, 2.6) for i in range(4)]
    for i, p in enumerate(m[:3]): commit(ax, *p, 'ABC'[i], BLUE)
    line(ax, m[0], m[2])
    if k == 0:
        commit(ax, *m[3], 'D', BLUE); line(ax, m[2], m[3])
        f = [(m[1][0] + 0.75, 1.3), (m[1][0] + 1.7, 1.3)]
        line(ax, m[1], f[0]); line(ax, f[0], f[1]); commit(ax, *f[0], 'E', GREEN); commit(ax, *f[1], 'F', GREEN)
        label(ax, m[3][0] + 0.75, 2.6, 'main', 9.5, '#2f6fd0', mono=True); label(ax, f[1][0] + 0.85, 1.3, 'feature', 9.5, '#2e9950', mono=True)
    elif k == 1:
        commit(ax, *m[3], 'D', BLUE); line(ax, m[2], m[3])
        f = [(m[1][0] + 0.75, 1.3), (m[1][0] + 1.7, 1.3)]
        mg = (m[3][0] + 0.95, 2.6)
        line(ax, m[1], f[0]); line(ax, f[0], f[1]); line(ax, f[1], mg); line(ax, m[3], mg)
        commit(ax, *f[0], 'E', GREEN); commit(ax, *f[1], 'F', GREEN); commit(ax, *mg, 'M', ORANGE)
        label(ax, mg[0], 3.15, 'main', 9.5, '#2f6fd0', mono=True); label(ax, x0 + 2.2, 0.55, '保留分叉历史，产生一个合并提交 M', 9.2, '#868e96')
    else:
        commit(ax, *m[3], 'D', BLUE); line(ax, m[2], m[3])
        f = [(m[3][0] + 0.95, 2.6), (m[3][0] + 1.9, 2.6)]
        line(ax, m[3], f[1]); commit(ax, *f[0], "E'", GREEN); commit(ax, *f[1], "F'", GREEN)
        label(ax, m[3][0], 3.15, 'main', 9.5, '#2f6fd0', mono=True); label(ax, f[1][0], 3.15, 'feature', 9.5, '#2e9950', mono=True)
        label(ax, x0 + 2.8, 0.55, "历史变成一条直线，E、F 被改写为新提交 E'、F'", 9.2, '#868e96')
for x in (4.85, 10.4): ax.plot([x, x], [0.3, 4.0], color='#dee2e6', lw=1)
save(fig, G, 'merge_rebase.png')

# ---- E. fork
fig, ax = canvas(11, 4.6, 0.85)
box(ax, 0.3, 3.0, 4.2, 1.2, '上游仓库 upstream\ncotes2020/jekyll-theme-chirpy', GREY, fs=9.5)
box(ax, 6.5, 3.0, 4.2, 1.2, '自己的仓库 origin\ndu-xinyi/du-xinyi.github.io', BLUE, fs=9.5)
box(ax, 3.4, 0.4, 4.2, 1.2, '本地仓库', GREEN, fs=10.5)
label(ax, 5.5, 4.4, 'GitHub', 9.5, '#868e96')
arrow(ax, (4.5, 3.6), (6.5, 3.6), 'fork（一次性复制）', off=(0, 0.22), ls='--')
arrow(ax, (2.4, 3.0), (4.4, 1.6), 'git fetch upstream\ngit merge upstream/master', off=(-1.75, 0.0), fs=9)
arrow(ax, (6.2, 1.6), (8.0, 3.0), 'git push', off=(-0.75, 0.05), fs=9)
arrow(ax, (9.2, 3.0), (7.3, 1.6), 'git clone / git pull', off=(1.35, -0.05), fs=9)
save(fig, G, 'fork_upstream.png')

# ---- K. openocd
fig, ax = canvas(13, 3.5, 0.85)
box(ax, 0.3, 2.2, 2.0, 0.8, 'GDB', BLUE); box(ax, 0.3, 1.2, 2.0, 0.8, 'Telnet', BLUE); box(ax, 0.3, 0.2, 2.0, 0.8, 'IDE / 脚本', BLUE)
box(ax, 4.3, 0.9, 2.2, 1.5, 'OpenOCD', ORANGE, fs=12, bold=True)
arrow(ax, (2.3, 2.6), (4.3, 2.0), 'TCP 3333', off=(0.1, 0.25), fs=9); arrow(ax, (2.3, 1.6), (4.3, 1.65), 'TCP 4444', off=(0, 0.17), fs=9); arrow(ax, (2.3, 0.6), (4.3, 1.25), 'TCP 6666', off=(0.15, -0.25), fs=9)
box(ax, 7.9, 0.9, 2.3, 1.5, '调试适配器\nST-Link / J-Link\nDAP-Link', GREEN, fs=9.5)
arrow(ax, (6.5, 1.65), (7.9, 1.65), 'USB', off=(0, 0.17), fs=9, style='<|-|>')
box(ax, 11.3, 0.9, 1.4, 1.5, '芯片\nMCU', PURPLE, fs=10.5)
arrow(ax, (10.2, 1.65), (11.3, 1.65), 'SWD\nJTAG', off=(0, 0.35), fs=9, style='<|-|>')
label(ax, 5.4, 0.55, 'interface/*.cfg  target/*.cfg', 8.2, '#868e96', mono=True)
label(ax, 3.4, 3.27, '电脑', 9.5, '#868e96'); ax.add_patch(Rectangle((0.1, 0.05), 6.6, 3.08, fill=False, ec='#ced4da', ls='--', lw=1))
save(fig, '2023-10-26-openocd-debugging', 'architecture.png')

# ---- M. race
fig, ax = canvas(10, 4.9, 0.85)
cols = [(0.3, 3.6, '线程A'), (4.0, 3.6, '线程B'), (7.7, 2.0, '内存中的 n')]
for x, w, t in cols: box(ax, x, 4.1, w, 0.6, t, GREY, fs=10.5, bold=True, round=0.05)
rows = [('读取 n，得到 0', '', '0'), ('', '读取 n，得到 0', '0'), ('计算 0 + 1', '', '0'), ('', '计算 0 + 1', '0'), ('写回 n = 1', '', '1'), ('', '写回 n = 1', '1')]
for i, (a, b, n) in enumerate(rows):
    y = 3.45 - i * 0.6
    if a: box(ax, 0.3, y, 3.6, 0.5, a, BLUE, fs=9.8, round=0.05)
    if b: box(ax, 4.0, y, 3.6, 0.5, b, GREEN, fs=9.8, round=0.05)
    box(ax, 7.7, y, 2.0, 0.5, n, RED if i >= 4 else ('#ffffff', '#ced4da'), fs=10.5, round=0.05)
label(ax, 5.0, 0.1, '两个线程各执行一次 n++，结果却是 1 而不是 2：线程B读到的是线程A写回之前的旧值', 9.5, '#d64545', va='bottom')
save(fig, '2023-10-12-cpp-multithreading', 'data_race.png')

# ---- N. mount
fig, ax = canvas(12, 4.6, 0.85)
label(ax, 2.2, 4.3, '设备 / 存储', 11, bold=True); label(ax, 8.9, 4.3, '目录树', 11, bold=True)
devs = [('/dev/nvme0n1p2\next4', BLUE), ('/dev/sdb1\nexfat', GREEN), ('192.168.1.10:/export/data\nnfs', ORANGE), ('tmpfs\n内存', PURPLE)]
tree = [('/', 0, BLUE), ('├── home', 0.0, None), ('├── etc', 0.0, None), ('└── mnt', 0.0, None), ('    ├── usb', 0, GREEN), ('    ├── nfs', 0, ORANGE), ('    └── ramdisk', 0, PURPLE)]
ty = [3.75 - i * 0.5 for i in range(len(tree))]
for (t, _, c), y in zip(tree, ty):
    if c: box(ax, 7.2, y - 0.2, 3.6, 0.4, '', c, round=0.05, lw=1.1)
    label(ax, 7.35, y, t, 10, ha='left', mono=True)
target = [0, 4, 5, 6]
for i, (t, c) in enumerate(devs):
    y = 3.25 - i * 0.95
    box(ax, 0.3, y, 3.8, 0.75, t, c, fs=9.2)
    arrow(ax, (4.1, y + 0.375), (7.2, ty[target[i]]), 'mount' if i == 0 else None, color=c[1], off=(0, 0.2), fs=9)
label(ax, 6.0, 0.25, '挂载 = 把一个文件系统接到目录树的某个目录（挂载点）上', 9.5, '#868e96')
save(fig, '2025-09-22-linux-mounting', 'mount_tree.png')

# ---- F. containers layout
fig, ax = canvas(13, 7.2, 0.8)
def cells(ax, x, y, vals, c, w=0.7, h=0.6, fs=10):
    for i, v in enumerate(vals): box(ax, x + i * w, y, w, h, v, c, fs=fs, round=0.02, lw=1.2)
label(ax, 0.3, 6.75, 'std::vector', 11.5, ha='left', bold=True, mono=True); label(ax, 3.0, 6.75, '一段连续内存，尾部预留容量', 9.5, '#868e96', ha='left')
cells(ax, 0.3, 5.85, list('12345'), BLUE); cells(ax, 3.8, 5.85, ['', '', ''], GREY)
ax.annotate('', (3.8, 5.7), (0.3, 5.7), arrowprops=dict(arrowstyle='<->', color='#868e96')); label(ax, 2.05, 5.5, 'size = 5', 9, '#868e96')
ax.annotate('', (5.9, 5.3), (0.3, 5.3), arrowprops=dict(arrowstyle='<->', color='#868e96')); label(ax, 3.1, 5.1, 'capacity = 8', 9, '#868e96')
label(ax, 7.0, 6.75, 'std::deque', 11.5, ha='left', bold=True, mono=True); label(ax, 9.5, 6.75, '多段连续内存 + 索引表', 9.5, '#868e96', ha='left')
cells(ax, 7.0, 5.0, ['', '', ''], ORANGE, w=0.6, h=0.5); label(ax, 7.9, 4.75, '索引表', 9, '#868e96')
for i, (y, vals, c) in enumerate([(6.0, ['', '', '1', '2'], BLUE), (5.25, list('3456'), BLUE), (4.5, ['7', '8', '', ''], BLUE)]):
    for j, v in enumerate(vals): box(ax, 9.6 + j * 0.7, y, 0.7, 0.55, v, c if v else GREY, fs=10, round=0.02, lw=1.2)
    arrow(ax, (7.3 + i * 0.6, 5.5 if i < 2 else 5.0), (9.6, y + 0.27), rad=0.0, lw=1.1)
label(ax, 0.3, 3.75, 'std::list', 11.5, ha='left', bold=True, mono=True); label(ax, 2.6, 3.75, '双向链表，每个节点单独分配，保存前后两个指针', 9.5, '#868e96', ha='left')
for i, v in enumerate('1234'):
    x = 0.6 + i * 2.6; box(ax, x, 2.75, 1.5, 0.6, '', BLUE, round=0.03, lw=1.2)
    ax.plot([x + 0.4, x + 0.4], [2.75, 3.35], color=BLUE[1], lw=1); ax.plot([x + 1.1, x + 1.1], [2.75, 3.35], color=BLUE[1], lw=1)
    label(ax, x + 0.2, 3.05, '←', 9, '#868e96'); label(ax, x + 0.75, 3.05, v, 10.5); label(ax, x + 1.3, 3.05, '→', 9, '#868e96')
    if i < 3: arrow(ax, (x + 1.5, 3.17), (x + 2.6, 3.17), lw=1.1); arrow(ax, (x + 2.6, 2.93), (x + 1.5, 2.93), lw=1.1)
label(ax, 0.3, 1.95, 'std::forward_list', 11.5, ha='left', bold=True, mono=True); label(ax, 4.4, 1.95, '单向链表，每个节点只保存后一个指针', 9.5, '#868e96', ha='left')
for i, v in enumerate('1234'):
    x = 0.6 + i * 2.6; box(ax, x, 0.95, 1.1, 0.6, '', GREEN, round=0.03, lw=1.2)
    ax.plot([x + 0.7, x + 0.7], [0.95, 1.55], color=GREEN[1], lw=1)
    label(ax, x + 0.35, 1.25, v, 10.5); label(ax, x + 0.9, 1.25, '→', 9, '#868e96')
    if i < 3: arrow(ax, (x + 1.1, 1.25), (x + 2.6, 1.25), lw=1.1)
label(ax, 6.5, 0.35, '蓝色、绿色为已有元素，灰色为已分配但尚未使用的空间', 9, '#868e96')
save(fig, '2023-10-21-cpp-containers', 'memory_layout.png')
