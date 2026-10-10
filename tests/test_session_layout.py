# -*- coding: utf-8 -*-
"""``h3_continuous/<剧本名>/<版本名>/`` 两层布局回归。

背景：产出以前只有一层 ``h3_continuous/<会话名>/``。同一部剧本重新渲一版会
**原地覆盖**上一版的成片，用户想对比两条 take 只能手抄目录名。现在按
「剧本 / 版本」分两层，版本名留空自动 ``r1``/``r2``…

这个文件守的是**改目录最容易踩、且踩了只会烧显卡**的四条：

  ① 升级前渲好的单层目录必须**原位沿用**。缓存键算的是内容，但段文件躺在
     哪儿是断点续渲的立足点；把它挪进两层布局 = 本轮看不见旧段 = 凭空全量重渲。
  ② ``fresh``（复用关掉）只能**另开一页**，绝不能改到已有版本的目录，也不能
     把同一轮里后面几段的目录改掉。
  ③ 目录名与成片名必须由**同一个表达式**推出：面板和清理都按「与目录同名的
     mp4」认成片，那里用 ``basename(dir)``、Session 里若用 ``sanitize(name)``，
     旧布局截胡时分叉，成片永远认不出来 → 被当成中间文件删掉。
  ④ 会话名归一化**只能有一套实现**（``session.sanitize``）：曾经 common 另抄
     一份正则、靠 import 期 monkey-patch 对齐，导入顺序一变就分叉成两个目录
     （渲染写 A、面板查 B → 断点渲染又从第 1 段开始）。

跑法（不需要 GPU，也不需要 ComfyUI 在跑）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_session_layout.py
"""
import json
import os
import shutil
import sys
import tempfile

# 本包在 <ComfyUI>/custom_nodes/ 下，所以上两级就是 ComfyUI 根目录。
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

import folder_paths  # noqa: E402

from ComfyUI_Marquee_Director.marquee_director import session as SESS  # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("  [FAIL]", msg)
    return cond


def _touch_session(path):
    """造一个「看起来渲过」的目录：有 manifest.json 就够。"""
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": SESS.MANIFEST_VERSION, "session": os.path.basename(path),
                   "segments": []}, f)
    return path


tmp = tempfile.mkdtemp(prefix="marquee_layout_")
try:
    out = os.path.join(tmp, "output")
    store = os.path.join(out, SESS.ROOT)
    # 全包唯一碰 folder_paths 的地方是 session.output_root()，所以换根只需要
    # 换这一个桩 —— 顺带验证了「没有第二处写死的输出根」。
    folder_paths.get_output_directory = lambda: out
    check(SESS.output_root() == out, "output_root() 没有走 folder_paths")
    check(SESS.store_root() == store, "store_root() 拼错了：%s" % SESS.store_root())

    # ------------------------------------------------- ③ 空名 / 只有一边的兜底
    d, s, v = SESS.resolve_session_dir("", "")
    check(d == os.path.join(store, SESS.UNNAMED) and s == "" and v == SESS.UNNAMED,
          "两个名字都没有时应落到 UNNAMED 单层目录，得到 %r/%r/%r" % (d, s, v))
    d, s, v = SESS.resolve_session_dir("take1", "")
    check(d == os.path.join(store, "take1") and s == "",
          "只有会话名时必须维持单层（不能套娃出 take1/take1），得到 %r" % d)

    # ------------------------------------------------- ② 两层 + 自动版本号
    d, s, v = SESS.resolve_session_dir("", script="曹贼的性价比")
    check(s == "曹贼的性价比" and v == "r1" and d == os.path.join(store, "曹贼的性价比", "r1"),
          "剧本已知、版本留空应自动 r1，得到 %r/%r/%r" % (d, s, v))
    _touch_session(d)
    d2, _, v2 = SESS.resolve_session_dir("", script="曹贼的性价比")
    check(d2 == d and v2 == "r1",
          "reuse 开着时必须接在最新一版上（断点续渲），得到 %r" % d2)
    d3, _, v3 = SESS.resolve_session_dir("", script="曹贼的性价比", fresh=True)
    check(d3 == os.path.join(store, "曹贼的性价比", "r2") and v3 == "r2",
          "fresh 应另开一页 r2，得到 %r/%r" % (d3, v3))

    # 名字撞车不能套娃：<剧本>/<剧本>/ 是无意义的第二层
    d, s, v = SESS.resolve_session_dir("同名词", script="同名词")
    check(v != s, "会话名 == 剧本名时不能生成 <剧本>/<剧本>/，得到 %r" % d)
    check(d == os.path.join(store, "同名词", v), "撞名后应落到 <剧本>/rN，得到 %r" % d)

    # 显式版本名照原样用（用户自己命名 take 的场景）
    d, s, v = SESS.resolve_session_dir("v2暖色", script="曹贼的性价比")
    check(s == "曹贼的性价比" and v == "v2暖色"
          and d == os.path.join(store, "曹贼的性价比", "v2暖色"),
          "显式版本名应被采用，得到 %r/%r" % (d, v))

    # ------------------------------------------------- ① 旧单层目录原位沿用
    legacy = _touch_session(os.path.join(store, "old_take"))
    d, s, v = SESS.resolve_session_dir("old_take", script="某剧本")
    check(d == legacy and v == "old_take" and s == "",
          "旧单层目录有产出时必须沿用（否则等于强制全量重渲），得到 %r" % d)
    # 但明确要重开一轮（fresh）就该走新布局
    d, _, _ = SESS.resolve_session_dir("old_take", script="某剧本", fresh=True)
    check(d.startswith(os.path.join(store, "某剧本")),
          "fresh=True 时旧目录不再截胡，应落到两层布局，得到 %r" % d)
    # 旧位置只有个空壳（没产出）→ 不算会话目录，让位给新布局
    os.makedirs(os.path.join(store, "empty_shell"), exist_ok=True)
    check(not SESS.is_session_dir(os.path.join(store, "empty_shell")),
          "空目录不该被认成会话（否则清理会把它当中间文件扫掉）")
    d, _, v = SESS.resolve_session_dir("empty_shell", script="某剧本")
    check(d == os.path.join(store, "某剧本", "empty_shell"),
          "空壳旧目录不该截胡，应落到两层布局（版本名本身要保留），得到 %r" % d)

    # 剧本夹本身不能被认成会话
    check(not SESS.is_session_dir(os.path.join(store, "曹贼的性价比")),
          "只含 r1/r2 子目录的剧本夹不是会话目录")

    # ------------------------------------------------- ③ Session 与目录同名成片
    sess = SESS.Session("v3冷色", script="曹贼的性价比")
    check(sess.dir == os.path.join(store, "曹贼的性价比", "v3冷色"),
          "Session.dir 与 resolve 不一致：%s" % sess.dir)
    check(sess.name == "v3冷色" and sess.script == "曹贼的性价比",
          "Session 名字/剧本归属不对：%r %r" % (sess.name, sess.script))
    check(os.path.basename(sess.dir) == sess.name,
          "成片名必须等于目录名（面板/清理按 <目录名>.mp4 认成片）")
    check(sess.joined_path() == os.path.join(sess.dir, "%s.mp4" % sess.name),
          "joined_path 不对：%s" % sess.joined_path())
    check(os.path.isdir(sess.dir), "Session 没有把目录建出来")

    # from_settings：所有构造点走同一个入口，剧本名那一层不能丢
    st = {"session_name": "v3冷色", "script_name": "曹贼的性价比"}
    check(SESS.Session.from_settings(st).dir == sess.dir,
          "from_settings 与直接构造得到不同目录（会分居两地）")
    check(SESS.Session.from_settings({"session_name": "v3冷色"}).dir != sess.dir,
          "漏传 script_name 时应能看出差别（这正是要靠 from_settings 统一的原因）")

    # from_dir：认领算好的目录，不重新解析（重算的每次机会都是一次分叉）
    adopted = SESS.Session.from_dir(sess.dir)
    check(adopted.dir == sess.dir and adopted.name == sess.name
          and adopted.script == sess.script,
          "from_dir 没有原样认领：%r %r %r" % (adopted.dir, adopted.name, adopted.script))

    # manifest 记名字、**绝不记绝对路径**
    sess.save([], {"notes": "x"})
    with open(sess.manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    check(man.get("script") == "曹贼的性价比", "manifest 没记剧本名：%r" % man.get("script"))
    check(tmp not in json.dumps(man, ensure_ascii=False),
          "manifest 里出现了绝对路径，换根以后会变成第二次分叉")

    # ------------------------------------------------- 遍历：两层 + 单层都要看见
    found = SESS.iter_session_dirs()
    def _rel(p):
        return os.path.relpath(p, store).replace("\\", "/")
    names = set(_rel(p) for p in found)
    check(_rel(legacy) in names, "老的单层会话没被扫到：%r" % names)
    check(_rel(sess.dir) in names, "两层会话没被扫到：%r" % names)
    check("曹贼的性价比" not in names, "剧本夹被当成会话扫进来了：%r" % names)
    check(SESS.script_of(sess.dir) == "曹贼的性价比", "script_of 两层判断错")
    check(SESS.script_of(legacy) == "", "script_of 单层应返回 ''")

    # ------------------------------------------------- ④ 归一化只有一套
    import importlib
    common = importlib.import_module(
        "ComfyUI_Marquee_Director.marquee_director.common")
    check(common.sanitize is SESS.sanitize,
          "common.sanitize 不是 session.sanitize —— 又出现第二套归一化，"
          "面板/渲染会分叉成两个目录")
    check(common.safe_session_name("", "  ", "曹贼 的 片名") == SESS.sanitize("曹贼 的 片名"),
          "safe_session_name 兜底链不对")
    check(common.safe_session_name("", "") == SESS.UNNAMED,
          "全空时必须落到全包唯一的兜底常量")
    check(SESS.sanitize("../../etc/passwd").find("..") < 0,
          "归一化必须拦住往上跳目录")

    print("\n两层目录回归: %s"
          % ("全部通过" if not FAIL else "%d 条失败" % len(FAIL)))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

sys.exit(1 if FAIL else 0)
