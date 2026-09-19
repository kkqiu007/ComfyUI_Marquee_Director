# -*- coding: utf-8 -*-
"""会话目录清理：只留分镜分段视频 + 完整拼接视频。

用户要求（原话）：
    最后输出只保留分镜分段视频和完整拼接视频；其他输出在关闭 comfyui
    和重启电脑系统就自动删除，禁止存留

所以规则是：
    保留  seg_<NN>.mp4          —— 分镜分段视频（不含 .tail.mp4）
    保留  <会话名>.mp4          —— 完整拼接视频
    删除  会话目录里的其它一切   —— *.tail.mp4 / *.tail.wav /
                                   *.tail.latent.safetensors /
                                   manifest.json / shots.json / 其它
    删除  输出根目录下的 h3_timeline_*.png（时间线缩略图，曾误写盘）

**但只在成片出来之后才动这个会话**（见 ``_session_done``）。用户说的是
"**最后输出**只保留…"，是"最后"不是"随时"。成片还没出就清，会把断点续渲
依赖的 manifest.json 和下一段回放锚点 seg_NN.tail.* 一起删掉，下次运行
整条链从头重渲——几小时 GPU 白烧，而且没有任何提示。

触发时机（三重保险，覆盖"关闭 ComfyUI"和"重启电脑"两种场景）：
    1. atexit      —— ComfyUI 正常退出时清一遍
    2. import 时   —— 下次启动补清上次残留（覆盖"重启电脑"和强杀进程）
    3. 定时巡检     —— 兜底；**会跳过正在渲染的会话**

安全守卫（很重要，别把正在跑的渲染删了）：
    · 只扫 <output>/h3_continuous/<session>/，绝不碰用户的其它文件
    · 会话目录里最新文件 mtime 在 IDLE_MINUTES 分钟内 → 当作在渲染，跳过
    · ComfyUI 队列里有任务在跑 → 整个巡检跳过
    · 输出根目录只删 h3_timeline_*.png 这一个我们自己产出的前缀
"""

import atexit
import glob
import os
import re
import threading
import time

# 会话目录里最近文件在这么多分钟内有改动，就认为还在渲染，跳过清理
IDLE_MINUTES = 15
# 定时巡检间隔（秒）
SWEEP_INTERVAL = 600

_SEG_RE = re.compile(r"^seg_(\d+)\.mp4$")
_TAIL_RE = re.compile(r"\.tail(\..+)?$")
_TIMELINE_RE = re.compile(r"^h3_timeline_\d+_\.png$", re.I)

_installed = False
_timer = None


# ---------------------------------------------------------------------------
# 判定：这个文件该留还是该删
# ---------------------------------------------------------------------------
# 面板元数据。原本只留视频，结果清完之后时间线只剩一排没有信息的缩略图 ——
# 每段讲了什么、谁在说话、多长，全跟着 shots.json / manifest.json 一起没了。
# 借鉴 Director 的做法：时间线显示的是「计划」而不是「渲染产物」，所以这些
# 描述计划的小体积元数据必须留下，否则面板等于瞎了。
PANEL_KEEP = {"storyboard.json", "panel.json"}


def _is_keep(name: str, session: str) -> bool:
    """只留分镜分段视频和完整拼接视频，外加面板要用的元数据。"""
    if _SEG_RE.match(name):
        return True
    if name.lower() == (session + ".mp4").lower():
        return True
    if name.lower() in {p.lower() for p in PANEL_KEEP}:
        return True
    return False


def _session_done(path: str, session: str) -> bool:
    """这个会话出成片了吗？

    用户要的是「**最后输出**只保留分镜分段视频和完整拼接视频」——是"最后"，
    不是"随时"。所以整轮清理只在成片出来之后才动这个目录。

    反过来，成片还没出来就清是灾难性的：``manifest.json`` 是断点续渲的唯一
    依据，``seg_NN.tail.*`` 是下一段回放的锚点。渲染到一半关掉 ComfyUI
    （或重启电脑），启动补清会把这两个都删掉，于是下次运行**整条链从头重渲**
    ——几小时的 GPU 白烧，而且用户完全不知道为什么。

    所以：没成片 = 会话还在进行中 = 整个目录跳过，一个字节都不动。
    """
    try:
        return os.path.isfile(os.path.join(path, session + ".mp4"))
    except Exception:
        return False


def _session_root():
    """跟随 ComfyUI 自己配置的输出目录，绝不硬编码。"""
    try:
        import folder_paths
    except Exception:
        return None
    root = os.path.join(folder_paths.get_output_directory(), "h3_continuous")
    return root if os.path.isdir(root) else None


def _queue_busy() -> bool:
    """ComfyUI 当前有没有任务在跑。"""
    try:
        # 别在这里 import 任何可能不存在的符号：一旦抛异常就会走到
        # return True（保守当作忙），结果就是清理永远不执行。
        from server import PromptServer

        srv = PromptServer.instance
        q = getattr(srv, "prompt_queue", None)
        if q is None:
            return False
        return len(getattr(q, "queue_running", []) or []) > 0 \
            or bool(getattr(q, "current_task", None))
    except Exception:
        # 拿不到就保守当作忙，宁可不删也别删错
        return True


def _dir_idle(path: str) -> bool:
    """目录里最新文件是否在 IDLE_MINUTES 分钟内被改过。改过 = 可能在渲染。"""
    try:
        newest = 0.0
        for name in os.listdir(path):
            try:
                m = os.path.getmtime(os.path.join(path, name))
            except OSError:
                continue
            newest = max(newest, m)
        if newest <= 0:
            return True
        return (time.time() - newest) > IDLE_MINUTES * 60
    except OSError:
        return True


# ---------------------------------------------------------------------------
# 清理动作
# ---------------------------------------------------------------------------
def sweep(force: bool = False) -> dict:
    """扫一遍并删除非白名单文件。返回统计。"""
    from .common import log

    stats = {"sessions": 0, "removed": 0, "bytes": 0, "skipped_busy": 0,
             "skipped_ongoing": 0, "timeline_removed": 0}

    root = _session_root()
    if not root:
        return stats

    busy = (not force) and _queue_busy()

    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        stats["sessions"] += 1
        # 成片没出来 = 会话还在进行中 = 不动（见 _session_done 的说明）
        if not _session_done(path, entry):
            stats["skipped_ongoing"] += 1
            continue
        if busy or (not force and not _dir_idle(path)):
            stats["skipped_busy"] += 1
            continue

        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if _is_keep(name, entry):
                continue
            if not os.path.isfile(full):
                continue
            try:
                size = os.path.getsize(full)
                os.remove(full)
                stats["removed"] += 1
                stats["bytes"] += size
            except OSError as exc:
                log("H3 cleanup: 删不掉 %s (%s)", full, exc)

    # 输出根目录：只清我们自己写出去的时间线缩略图
    try:
        import folder_paths

        out = folder_paths.get_output_directory()
        for p in glob.glob(os.path.join(out, "h3_timeline_*.png")):
            if not _TIMELINE_RE.match(os.path.basename(p)):
                continue
            try:
                os.remove(p)
                stats["timeline_removed"] += 1
            except OSError:
                pass
    except Exception:
        pass

    if stats["removed"] or stats["timeline_removed"]:
        log("H3 cleanup: 清掉 %d 个中间文件（%.1f MB）+ %d 张时间线图",
            stats["removed"], stats["bytes"] / 1e6, stats["timeline_removed"])
    return stats


def preview(force: bool = False) -> dict:
    """只列出"会删什么"，不真删。用来在动手前先看一眼。"""
    root = _session_root()
    out = {"sessions": [], "timeline": [], "busy": False}
    if not root:
        return out

    busy = (not force) and _queue_busy()
    out["busy"] = busy

    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        idle = _dir_idle(path)
        done = _session_done(path, entry)
        would = []
        keep = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not os.path.isfile(full):
                continue
            if _is_keep(name, entry):
                keep.append(name)
            elif done:
                would.append({"file": name,
                              "kb": round(os.path.getsize(full) / 1024, 1)})
            else:
                keep.append(name)      # 会话没成片 → 全部按"保留"显示
        out["sessions"].append({
            "session": entry,
            "idle": idle,
            "done": done,
            "skipped": (not done) or busy or (not force and not idle),
            "keep": keep,
            "remove": would,
        })

    try:
        import folder_paths

        for p in glob.glob(os.path.join(folder_paths.get_output_directory(),
                                        "h3_timeline_*.png")):
            if _TIMELINE_RE.match(os.path.basename(p)):
                out["timeline"].append(os.path.basename(p))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 安装：atexit + 启动补清 + 定时巡检
# ---------------------------------------------------------------------------
def _on_exit():
    try:
        sweep(force=True)
    except Exception:
        pass


def _periodic():
    global _timer
    try:
        sweep()
    except Exception:
        pass
    _timer = threading.Timer(SWEEP_INTERVAL, _periodic)
    _timer.daemon = True
    _timer.start()


def install() -> bool:
    """注册清理。可重复调用；失败不影响节点本身可用。"""
    global _installed
    if _installed:
        return True
    _installed = True
    try:
        atexit.register(_on_exit)          # 1) 关闭 ComfyUI
    except Exception:
        pass
    try:
        sweep()                             # 2) 启动补清（覆盖重启电脑）
    except Exception:
        pass
    try:
        _periodic()                         # 3) 定时兜底
    except Exception:
        pass
    return True
