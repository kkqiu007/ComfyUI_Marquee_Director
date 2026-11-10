# -*- coding: utf-8 -*-
"""2026-10-17 自检：路径安全（会话名净化 + 导演包解压）。

为什么单独一个文件：这两处是**唯一**把外部输入（控件文本、PACK 里的 Project 名、
上传的 zip）变成磁盘路径的地方。它们的失败方式都不是报错，而是写到别处、
或者目录根本没建出来。

守五条：
  ① 路径穿越：``../``、``..\\``、``/abs``、``C:/`` 都不能带出根目录；
  ② Windows 设备名：CON / NUL / COM1 / LPT1 / aux / PRN 及其带扩展名形式，
     净化后必须**真的能建目录并写进文件** —— ``NUL`` 的坑是 makedirs 不报错、
     目录却不存在，之后每次写段都 FileNotFoundError；
  ③ 幂等：``sanitize(sanitize(x)) == sanitize(x)``。会话名会在解析器、Director、
     面板路由三处各过一遍，不幂等就会算出两个目录（历史故障：面板 done=0、
     断点续渲每次从第 1 段重来）；
  ④ 中文片名必须原样保留（磁盘上的文件夹名就是人看得懂的片名）；
  ⑤ zip-slip：恶意成员一律拒绝，正常成员照常解出，扩展名白名单生效。

跑法（必须用 ComfyUI 的 venv）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_path_safety.py
"""
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from marquee_director import pack as P                                # noqa: E402
from marquee_director.session import UNNAMED, sanitize               # noqa: E402

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg)


HOSTILE = [
    "../evil", "../../etc/passwd", "..\\..\\win", "/abs/path", "C:/windows",
    "a/../../b", "..", ".", "", "   ",
    "CON", "NUL", "COM1", "LPT1", "aux", "PRN",
    "CON.txt", "nul.png", "com9.log", "lpt1.mp4",
]

NORMAL = ["曹贼的性价比_v18_en_v10", "normal_name", "r1", "2026-10-17",
          "shot-list.v2", "a" * 200]


def test_traversal():
    """① 净化结果不含分隔符，也不含 .."""
    for raw in HOSTILE:
        got = sanitize(raw)
        check("/" not in got and "\\" not in got, "分隔符残留：%r -> %r" % (raw, got))
        check(".." not in got, "仍含 ..：%r -> %r" % (raw, got))
        check(got not in ("", ".", ".."), "净化成空/点：%r -> %r" % (raw, got))
    print("PASS ① 路径穿越被切断")


def test_reserved_names():
    """② 设备名带前缀，且真的能建目录 + 写文件。"""
    root = tempfile.mkdtemp(prefix="h3path_")
    try:
        for raw in HOSTILE:
            got = sanitize(raw)
            base = got.split(".", 1)[0].upper()
            check(base not in {"CON", "PRN", "AUX", "NUL"}
                  and not (len(base) == 4 and base[:3] in ("COM", "LPT")
                           and base[3].isdigit()),
                  "设备名没被处理：%r -> %r" % (raw, got))
            d = os.path.join(root, got)
            try:
                os.makedirs(d, exist_ok=True)
                check(os.path.isdir(d), "目录没建出来：%r -> %r" % (raw, d))
                with open(os.path.join(d, "seg_00.mp4"), "wb") as f:
                    f.write(b"x")
            except Exception as exc:
                check(False, "%r 建/写失败 %s: %s" % (raw, type(exc).__name__, exc))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("PASS ② 设备名已规避且真实可写")


def test_idempotent():
    """③ 幂等：再净化一次结果不变。"""
    for raw in HOSTILE + NORMAL:
        once = sanitize(raw)
        twice = sanitize(once)
        check(once == twice, "不幂等：%r -> %r -> %r" % (raw, once, twice))
    print("PASS ③ sanitize 幂等")


def test_cjk_kept():
    """④ 中文片名原样保留。"""
    for raw in ("曹贼的性价比_v18_en_v10", "第一集_开场", "测试"):
        check(sanitize(raw) == raw, "中文被改写：%r -> %r" % (raw, sanitize(raw)))
    check(sanitize("") == UNNAMED, "空名没落到 UNNAMED")
    check(len(sanitize("a" * 200)) <= 61, "超长名没截断")
    print("PASS ④ 中文保留 / 空名兜底 / 超长截断")


def _zip_with(members):
    tmp = Path(tempfile.mkdtemp(prefix="h3zip_"))
    z = tmp / "t.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return tmp, z


def test_zip_slip():
    """⑤ zip-slip 拒绝、正常成员接受、扩展名白名单生效。"""
    bad = [
        ("../evil.png", b"x"), ("/abs/evil.png", b"x"), ("C:/evil.png", b"x"),
        ("..\\evil.png", b"x"), ("refs/../evil.png", b"x"),
        ("refs/evil.exe", b"x"), ("refs/evil.png.exe", b"x"),
    ]
    for name, data in bad:
        tmp, z = _zip_with([(name, data)])
        try:
            rejected = False
            try:
                P.extract_pack_zip(z, tmp / "out")
            except P.PackError:
                rejected = True
            check(rejected, "恶意成员没被拒绝：%r" % name)
            # 越界产物绝不能出现在 tmp 之外
            escaped = [p for p in Path(tmp).parent.rglob("evil.*")
                       if str(p).startswith(str(Path(tmp).parent)) and tmp not in p.parents]
            check(not escaped, "文件越界落盘：%r" % escaped[:2])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    tmp, z = _zip_with([("refs/Picture1.png", b"x"), ("storyboard.json", b"{}")])
    try:
        dest = tmp / "out"
        P.extract_pack_zip(z, dest)
        check((dest / "refs" / "Picture1.png").is_file(), "正常成员没解出来")
        check((dest / "storyboard.json").is_file(), "正常 JSON 没解出来")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS ⑤ zip-slip 拒绝 / 正常成员通过")


if __name__ == "__main__":
    test_traversal()
    test_reserved_names()
    test_idempotent()
    test_cjk_kept()
    test_zip_slip()
    if FAILS:
        print("\n发现 %d 处路径安全问题：" % len(FAILS))
        for f in FAILS:
            print("  - " + f)
        sys.exit(1)
    print("\n全部通过：会话名净化与导演包解压都拦得住。")
