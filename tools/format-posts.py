"""统一 _posts 下技术文章的写法格式。

用法：
    python3 tools/format-posts.py            # 处理全部技术文章
    python3 tools/format-posts.py _posts/x.md  # 只处理指定的文章

做的事情（重复运行结果不变）：
  - 表格全部左对齐并按显示宽度补齐（超过 60 宽的列不补齐）
  - 选项、参数这类“名称 说明”形式的列表转成表格
  - 代码块标签统一为紧跟在代码块下面的 {: file='...'}，代码块必须带语言
  - 中文与英文、数字之间加空格，中文与行内代码之间不加
  - 中文语境中的半角括号、冒号、逗号改为全角，段落、列表项、单元格末尾不带句号
  - 标题、代码块、表格前后各留一个空行
菜谱（_recipes）和 cookbook、first-post 两篇不处理。
"""
import re, sys, glob, os

HAN = r'一-鿿㐀-䶿'
WIDE = [(0x1100, 0x115f), (0x2e80, 0x303e), (0x3041, 0x33ff), (0x3400, 0x4dbf), (0x4e00, 0x9fff), (0xa000, 0xa4cf),
        (0xac00, 0xd7a3), (0xf900, 0xfaff), (0xfe30, 0xfe4f), (0xff00, 0xff60), (0xffe0, 0xffe6), (0x2460, 0x24ff)]
def width(t): return sum(2 if any(lo <= ord(ch) <= hi for lo, hi in WIDE) else 1 for ch in t)

C0, C1, O0, O1 = '', '', '', ''   # code / other placeholders
PROTECT = re.compile(r'(?P<code>``.+?``|`[^`\n]+`)|(?P<other>\$[^$\n]+\$|\]\([^)\n]*\)|<[A-Za-z/!][^>\n]*>|\{:[^}\n]*\}|https?://[^\s)）]+)')
NOUNS = {'ubuntu': 'Ubuntu', 'linux': 'Linux', 'windows': 'Windows', 'python': 'Python', 'keil': 'Keil', 'lambda': 'Lambda',
         'github': 'GitHub', 'sobel': 'Sobel', 'canny': 'Canny', 'laplacian': 'Laplacian', 'opencv': 'OpenCV'}

def prose(t, nouns=True):
    store = []
    def keep(m):
        store.append(m.group(0)); i = len(store) - 1
        return f'{C0}{i}{C1}' if m.group('code') else f'{O0}{i}{O1}'
    t = PROTECT.sub(keep, t)
    if nouns:
        for k, v in NOUNS.items():
            t = re.sub(rf'(?<![A-Za-z0-9_./\-]){k}(?![A-Za-z0-9_./\-])', v, t)
    # half-width punctuation next to Chinese -> full-width
    for _ in range(3):
        t = re.sub(rf'(?<=[{HAN}{C1}{O1}])\(([^()（）\n]*)\)', r'（\1）', t)
        t = re.sub(rf'(?<!\])\(([^()（）\n]*[{HAN}][^()（）\n]*)\)', r'（\1）', t)
    t = re.sub(rf'(?<=[{HAN}{C1}{O1}）]):(?=\s|$|[{HAN}{C0}{O0}])', '：', t)
    t = re.sub(rf'(?<=[{HAN}]),\s?(?=[{HAN}{C0}])', '，', t)
    # no spaces around full-width punctuation
    t = re.sub(r'(?<=[，。；：、！？（）]) +(?=\S)', '', t)
    t = re.sub(r'(?<=\S) +(?=[，。；：、！？）])', '', t)
    # a space between Chinese and Latin letters / digits
    t = re.sub(rf'(?<=[{HAN}])(?=[A-Za-z0-9])', ' ', t)
    t = re.sub(rf'(?<=[A-Za-z0-9%+)])(?=[{HAN}])', ' ', t)
    # no space between Chinese and inline code
    t = re.sub(rf'(?<=[{HAN}]) +(?={C0})', '', t)
    t = re.sub(rf'(?<={C1}) +(?=[{HAN}])', '', t)
    return re.sub(rf'[{C0}{O0}](\d+)[{C1}{O1}]', lambda m: store[int(m.group(1))], t)

def strip_period(t):
    return re.sub(r'。(\s*)$', r'\1', t)

def split_row(line):
    store = []
    def keep(m): store.append(m.group(0)); return f'{C0}{len(store) - 1}{C1}'
    t = re.sub(r'``.+?``|`[^`\n]+`|\$[^$\n]+\$', keep, line.strip())
    t = re.sub(r'^\||\|$', '', t)
    return [re.sub(rf'{C0}(\d+){C1}', lambda m: store[int(m.group(1))], c).strip() for c in t.split('|')]

def format_table(block):
    indent = re.match(r'\s*', block[0]).group(0)
    rows = [split_row(l) for l in block]
    n = len(rows[0])
    if len(block) < 3 or any(len(r) != n for r in rows) or not all(re.fullmatch(r':?-+:?', c) for c in rows[1]):
        print('   ! table left untouched:', block[0][:60]); return block
    seps = rows[1]
    body = [[strip_period(prose(c)) for c in r] for r in [rows[0]] + rows[2:]]
    align = []
    for i, s in enumerate(seps):
        centered = s.startswith(':') and s.endswith(':')
        if centered and n > 3: align.append('c')
        else: align.append('l')
    w = [max(3, *(width(r[i]) for r in body)) for i in range(n)]
    w = [x if x <= 60 else 0 for x in w]          # very long columns are left unpadded
    def cell(text, i):
        pad = max(0, w[i] - width(text))
        if align[i] == 'r': return ' ' * pad + text
        if align[i] == 'c': return ' ' * (pad // 2) + text + ' ' * (pad - pad // 2)
        return text + ' ' * pad
    def sep(i):
        k = w[i] or 4
        return {'l': ':' + '-' * (k - 1), 'r': '-' * (k - 1) + ':', 'c': ':' + '-' * (k - 2) + ':'}[align[i]]
    out = ['| ' + ' | '.join(cell(c, i) for i, c in enumerate(body[0])) + ' |',
           '| ' + ' | '.join(sep(i) for i in range(n)) + ' |']
    out += ['| ' + ' | '.join(cell(c, i) for i, c in enumerate(r)) + ' |' for r in body[1:]]
    return [indent + l for l in out]

CODE = r'``.+?``|`[^`\n]+`'
HEADERS = {'`tcp://`': '协议', '`std::set`': '容器', '`std::unordered_set`、`std::unordered_multiset`': '容器', '`[=, a]`': '写法',
           '`std::ref`': '函数', '`std::shared_lock`': '类型'}
REPORT = []

def split_item(text):
    """Split 'name desc' of an option-like list item; None when it is ordinary prose."""
    m = re.match(rf'^((?:{CODE})(?:(?:\s*[/+,，、或]\s*|\s+)(?:{CODE}))*)(\s*[:：]\s*|\s+)(\S.*)$', text)
    if m and re.match(rf'[{HAN}（(A-Za-z0-9]', m.group(3)) and not re.match(CODE, m.group(3)):
        return m.group(1), m.group(3)
    m = re.match(rf'^([\x21-\x7e][\x20-\x7e，、]*?)(\s*[:：]\s*|\s+)([{HAN}（].*)$', text)
    ident = r'(?:-{1,2}[\w=\-"./:]+|\[\w+\]|\*{0,2}[a-z_]\w*(?:\(\))?|[A-Z_]{2,})'
    if m and re.fullmatch(rf'{ident}(?:\s*[,，、]\s*{ident})*', m.group(1).strip()):
        return m.group(1).strip(), m.group(3)
    return None

def code_name(name):
    if '`' in name: return name
    return '、'.join(f'`{p.strip()}`' for p in re.split(r'[,，、]', name) if p.strip())

def convert_list(block):
    items = []                      # [text, [children]]
    for l in block:
        m = re.match(r'^- (.*)$', l)
        c = re.match(r'^\s+- (.*)$', l)
        if m: items.append([m.group(1).rstrip(), []])
        elif c: items[-1][1].append(c.group(1).rstrip())
        else:
            t = l.strip()
            if items[-1][1]: items[-1][1][-1] += t
            else: items[-1][0] += t
    # OpenCV style: 第N个参数，Type类型的`name`，说明
    cv = [re.match(r'^第.{1,2}个参数，\s*(.+?)\s*类型的\s*`(\w+)`\s*[，,]?\s*(.*)$', it[0]) for it in items]
    if all(cv) and len(items) >= 2:
        rows = ['| 参数 | 类型 | 说明 |', '| :--- | :--- | ---: |']
        for m, it in zip(cv, items):
            desc = '<br>'.join([m.group(3)] + it[1])
            rows.append(f'| `{m.group(2)}` | `{m.group(1)}` | {desc} |')
        return rows
    rows = []
    for text, children in items:
        sp = split_item(text)
        kids = [split_item(c) for c in children]
        if sp is None:
            if children and all(kids) and re.fullmatch(r'\[?[A-Za-z*]+\]?', text):   # group header such as [options]
                rows += [(code_name(n), d) for n, d in kids]; continue
            return None
        name, desc = sp
        if children:
            if not all(kids): return None
            desc = '<br>'.join([desc] + [f'{code_name(n)}：{d}' for n, d in kids])
        rows.append((code_name(name), desc))
    if len(rows) < 2 or all(r[0].startswith('`[') and not r[0].startswith('`[options') for r in rows): return None
    names = [r[0] for r in rows]
    if names[0] in HEADERS: head = HEADERS[names[0]]
    elif any('Ctrl' in n for n in names): head = '快捷键'
    elif all(n.endswith('()`') for n in names): head = '函数'
    elif all(re.match(r"`['\"#(0-9$]|`None", n) for n in names): head = '取值'
    else: head = '参数'
    tail = {'快捷键': '功能', '函数': '作用', '取值': '含义'}.get(head, '说明')
    return [f'| {head} | {tail} |', '| :--- | ---: |'] + [f'| {n} | {d} |' for n, d in rows]

def lists_to_tables(lines):
    out, i, fence = [], 0, False
    while i < len(lines):
        l = lines[i]
        if re.match(r'^\s*```', l): fence = not fence
        if fence or not l.startswith('- '):
            out.append(l); i += 1; continue
        j = i
        while j < len(lines) and (lines[j].startswith('- ') or re.match(r'^\s+\S', lines[j])): j += 1
        block = lines[i:j]
        bad = any(re.match(r'^\s+(```|\||\$\$)', b) or b.endswith('  ') for b in block)
        table = None if bad else convert_list(block)
        if table:
            REPORT.append((table[0], len(table) - 2, table[2][:70])); out += table
        else: out += block
        i = j
    return out

LANG = {'': 'text', 'shell': 'bash', 'zsh': 'bash', 'C++': 'c++'}
IAL = re.compile(r'^(\s*)\{:\s*file=([\'"])(.*)\2\s*\}\s*$')
STRUCT = re.compile(r'^\s*(#{1,6} |[-*+] |\d+\. |```|\||\{:|!\[|\$\$|>)')

def normalize(text):
    m = re.match(r'^---\n(.*?)\n---\n', text, re.S)
    front, body = m.group(0), text[m.end():]
    front = re.sub(r'^math: True$', 'math: true', front, flags=re.M)
    lines = lists_to_tables(body.split('\n'))
    out, i, fence, math = [], 0, None, False
    while i < len(lines):
        l = lines[i]
        fm = re.match(r'^(\s*)```(.*)$', l)
        if fence is not None:
            out.append(l)
            if fm and fm.group(2).strip() == '': fence = None
            i += 1; continue
        if fm:
            lang = fm.group(2).strip()
            out.append(f'{fm.group(1)}```{LANG.get(lang, lang)}'); fence = lang; i += 1; continue
        if l.strip() == '$$':
            math = not math; out.append(l.rstrip()); i += 1; continue
        if math:
            out.append(l); i += 1; continue
        im = IAL.match(l)
        if im:
            lab = prose(im.group(3)); q = '"' if "'" in lab else "'"
            out.append(f'{im.group(1)}{{: file={q}{lab}{q}}}'); i += 1; continue
        if re.match(r'^\s*\|', l):
            j = i
            while j < len(lines) and re.match(r'^\s*\|', lines[j]): j += 1
            out += format_table(lines[i:j]); i = j; continue
        hard = l.endswith('  ') and l.strip() != ''
        t = prose(l.rstrip())
        nxt = lines[i + 1] if i + 1 < len(lines) else ''
        if hard or nxt.strip() == '' or STRUCT.match(nxt) or STRUCT.match(l) and STRUCT.match(nxt):
            t = strip_period(t)
        out.append(t + ('  ' if hard else '')); i += 1
    # blank lines around headings, fences (+ their label) and tables
    res, fence = [], False
    def blank():
        if res and res[-1].strip() != '': res.append('')
    for k, l in enumerate(out):
        top = not l.startswith((' ', '\t'))
        prev = res[-1] if res else ''
        is_f = bool(re.match(r'^```', l))
        if fence:
            res.append(l)
            if is_f: fence = False
            continue
        if l.strip() == '' and prev.strip() == '' and res: continue
        if top and (re.match(r'^#{1,6} ', l) or is_f or (l.startswith('|') and not prev.startswith('|'))): blank()
        if top and l.strip() and re.match(r'^#{1,6} ', prev): blank()
        if top and l.strip() and not IAL.match(l) and (prev == '```' or IAL.match(prev) or re.match(r'^#{1,6} ', prev) or (prev.startswith('|') and not l.startswith('|'))): blank()
        res.append(l)
        if is_f: fence = True
    return front + '\n'.join(res).rstrip('\n') + '\n'

if __name__ == '__main__':
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    for p in (sys.argv[1:] or sorted(glob.glob('_posts/*.md'))):
        if 'cookbook' in p or 'first-post' in p: continue
        REPORT.clear(); s = open(p, encoding='utf8').read(); t = normalize(s); rep = list(REPORT)
        assert normalize(t) == t, ('not idempotent', p)
        if t != s:
            open(p, 'w', encoding='utf8').write(t)
            print(os.path.basename(p), 'tables from lists:', len(rep)); [print('     ', r) for r in rep]
            print(os.path.basename(p), 'changed lines:', sum(a != b for a, b in zip(s.split('\n'), t.split('\n'))), 'len', len(s.split('\n')), '->', len(t.split('\n')))
