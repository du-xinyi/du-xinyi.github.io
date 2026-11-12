"""重新运行一篇文章里的 Python 代码，检查输出与文中是否一致。

用法：
    python3 tools/check-post-code.py pytorch-cnn
    python3 tools/check-post-code.py pytorch-training --python ~/venv/bin/python --threads 4

规则：
  - 带有 import 的代码块当作一个完整的程序单独运行；没有 import 的代码块接在它
    前面的程序后面一起运行
  - 运行时提示某个名字未定义的代码块，从近到远尝试接在定义了这个名字的代码块后面运行，
    取第一个结果与文中一致的
  - 代码块后面紧跟着的 text 代码块视为这段程序的输出，逐行在实际输出中查找
  - print(...)  # 值 这样的行内注释，值必须出现在实际输出中
  - 含有耗时（秒、ms）的输出行每次运行都不同，不参与比较
需要特定文件或 GPU 的代码块会报告为失败，需要人工判断。
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('slug', help='文章文件名中日期之后的部分，如 pytorch-cnn')
parser.add_argument('--python', default=sys.executable, help='用来运行代码的 Python 解释器')
parser.add_argument('--threads', default='4', help='torch 使用的线程数，输出与线程数有关的文章需要固定')
parser.add_argument('--timeout', type=int, default=900)
args = parser.parse_args()

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
posts = [p for p in os.listdir(os.path.join(root, '_posts')) if p.endswith(args.slug + '.md')]
if len(posts) != 1:
    sys.exit(f'找不到唯一的文章：{args.slug}')
text = open(os.path.join(root, '_posts', posts[0]), encoding='utf8').read()

parts = re.findall(r'```(\w+)\n(.*?)```', text, re.S)
python_blocks = [(i, code) for i, (lang, code) in enumerate(parts) if lang == 'python']
prelude = f'import torch\ntorch.set_num_threads({int(args.threads)})\n'
ok = bad = skipped = 0
previous = ''
history = []  # 已经成功运行过的 (代码块, 完整程序)


workdir = tempfile.mkdtemp(prefix='check-post-code-')  # 代码保存的文件放在临时目录，不弄乱仓库


def run(program):
    return subprocess.run([os.path.expanduser(args.python), '-W', 'ignore', '-c', prelude + program],
                          capture_output=True, text=True, timeout=args.timeout, cwd=workdir)


def check(index, code, result):
    """返回这次运行与文中不一致的地方"""
    problems = []
    if result.returncode != 0:
        problems.append('运行失败：' + result.stderr.strip().split('\n')[-1][:160])
    if index + 1 < len(parts) and parts[index + 1][0] == 'text':
        for line in parts[index + 1][1].strip().split('\n'):
            line = line.strip()
            if line and line not in result.stdout and not re.search(r'\d+\.\d+ ?(秒|s|ms)|…|\.\.\.', line):
                problems.append('输出中没有这一行：' + line[:70])
                break
    for value in re.findall(r'print\(.*\)\s*#\s*(torch\.Size\(\[[^\]]*\]\)|True|False|None|-?\d+(?:\.\d+)?)\s*$', code, re.M):
        if value not in result.stdout:
            problems.append('注释中的值没有出现在输出中：' + value)
            break
    return problems


for number, (index, code) in enumerate(python_blocks):
    complete = bool(re.search(r'^(import |from )', code, re.M))
    program = code if complete else previous + '\n' + code
    try:
        result = run(program)
        problems = check(index, code, result)
        missing = re.search(r"NameError: name '(\w+)' is not defined", result.stderr)
        if result.returncode != 0 and missing:
            # 依赖前面某个代码块中的定义：从近到远尝试每个定义了这个名字的代码块，
            # 取第一个运行结果与文中一致的
            name = missing.group(1)
            for earlier_code, earlier_program in reversed(history):
                if re.search(rf'^(class|def) {name}\b|^{name} = ', earlier_code, re.M):
                    candidate = earlier_program + '\n' + code
                    candidate_result = run(candidate)
                    candidate_problems = check(index, code, candidate_result)
                    if candidate_result.returncode == 0 and (result.returncode != 0 or not candidate_problems):
                        program, result, problems = candidate, candidate_result, candidate_problems
                    if not candidate_problems:
                        break
    except subprocess.TimeoutExpired:
        bad += 1
        print(f'  代码块 {number}: 超时')
        continue
    if result.returncode != 0 and not complete:
        skipped += 1
        continue
    if result.returncode == 0:
        history.append((code, program))
    previous = program
    if problems:
        bad += 1
        print(f'  代码块 {number} ({code.strip().splitlines()[0][:50]!r}): {problems[0]}')
    else:
        ok += 1
print(f'{posts[0]}: 一致 {ok}，不一致 {bad}，无法单独运行 {skipped}，共 {len(python_blocks)} 个 Python 代码块')
shutil.rmtree(workdir, ignore_errors=True)
sys.exit(1 if bad else 0)
