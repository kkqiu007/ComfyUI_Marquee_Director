"""Frame maths, audio slicing, and the core MiniMax H3 nodes this pack drives.

Nothing here reimplements H3 conditioning. The chain calls the core nodes
(``MiniMaxH3ReferenceToVideo``, ``MiniMaxH3AddGuide``) and the core sampler, so a
ComfyUI update carries straight through instead of silently diverging.
"""

import logging

import torch

FPS = 24
AUDIO_LATENT_FPS = 40

try:
    from comfy_extras.nodes_minimax_h3 import (  # noqa: F401
        MiniMaxH3AddGuide,
        MiniMaxH3ReferenceToVideo,
        video_latent_t,
    )
except ImportError as exc:  # pragma: no cover - install-time guard
    raise ImportError(
        "ComfyUI_Marquee_Director needs a ComfyUI with MiniMax H3 support "
        "(comfy_extras/nodes_minimax_h3.py, ComfyUI 0.34.0 or newer). "
        "Update ComfyUI and restart."
    ) from exc

# One t-unit of the packed sequence per audio latent frame, 5/3 per pixel frame.
# The guide's audio and its picture have to be placed on that shared axis, which is
# the only reason this constant leaves the model file.
from comfy.ldm.minimax.model import FRAME_RESCALE  # noqa: E402


import re as _re

# 会话名保留中日韩 / 字母数字 / . -，其余压成 _。
# 这是**全包唯一的会话名归一化实现**，session.sanitize 直接复用它（见文件末尾
# 的 _install_session_sanitize_alias）。以前这里和 session.py 各有一套正则：
# 这一套留 CJK，session.py 那套 ``[^A-Za-z0-9._-]`` 把 CJK 全压成下划线 ——
# 于是同一个 "曹贼的性价比_v18_en_v10" 在这一侧是原样，在落盘侧变成
# "v18_en_v10"，面板查的目录和渲染写的目录是两个不同的根。
_SAFE_NAME_RE = _re.compile(r"[^\w\u4e00-\u9fff.-]+")


def safe_session_name(*candidates):
    """会话名（也是落盘的文件夹名）：保留中日韩 / 字母数字 / . -，其余压成 _。

    放在 common 是因为**同一个会话名会被五处用到**，必须算出同一个结果：
      1. H3PromptPackParser —— 解析 PACK 时定下会话名，写进 shots_json；
      2. H3Director —— 拿它当 chain 的 session（``settings["session_name"]``）；
      3. H3RepairSegmentNode / H3ExtendSession —— 修复/续接时定位会话目录；
      4. /h3/pack_preview + /h3/session —— 面板拿它去查「这个会话渲到第几段」；
      5. session.Session 落盘目录名。

    ★ 这一侧**保留 CJK 是刻意的**：PACK 头部的 Project 常常就是中文片名
      （「曹贼的性价比」），保留它=磁盘上那个目录名就是人看得懂的片名。真正
      的危险是"两侧算法不一致"——那会让渲染写 A 目录、面板查 B 目录，表现为
      「断点渲染又从第 1 段开始」以及修复节点抛
      ``segment 1 has no handoff clip``（因为从没渲过第 1 段，自然没有 tail）。
      所以 session.sanitize 必须与本函数同源，见下方的 monkey-patch 说明。

    候选顺序**必须**与 pack_nodes / routes 里的调用点一致（控件 → Project），
    否则解析器和预览会算出两个名字。
    """
    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        cleaned = _SAFE_NAME_RE.sub("_", text).strip("_.")
        if cleaned:
            return cleaned[:60]
    return "h3_pack"


def _install_session_sanitize_alias():
    """把 ``session.sanitize`` 指到本函数，保证"落盘名"与"查找名"同源。

    为什么不是直接改 session.py：session.sanitize 被 Session.__init__ 在
    **每次构造实例时**调用，而它原本的正则 ``[^A-Za-z0-9._-]`` 会丢 CJK。
    只改一处、保留两处的写法迟早会再次跑偏；这里做的是"单一实现 + 别名"，
    从根上消掉分歧。

    单独调用（离线测试）也安全：导入 session 失败时静默跳过，不阻断 common。
    """
    try:
        from . import session as _session_mod
    except Exception:                                            # pragma: no cover
        try:
            import session as _session_mod                       # 直接 / 独立加载
        except Exception:
            return
    if getattr(_session_mod, "_SANITIZE_IS_ALIAS", False):
        return
    _session_mod.sanitize = safe_session_name
    _session_mod._SANITIZE_IS_ALIAS = True


_install_session_sanitize_alias()


def log(msg, *args):
    logging.info("[H3 Continuous] " + msg, *args)


_CLIP_GUIDES = None


def _probe_clip_guides():
    """Does this core reserve a guide latent's real length in the packed sequence?

    ``PackedLayout`` is asked for a two-frame guide and its answer is counted. A
    core with multi-frame guide support reserves ``vt * frame_rows`` condition
    rows; the one that shipped before ``MiniMaxH3AddGuide`` reserves ``frame_rows``
    -- one frame -- whatever length of guide it is handed.

    A probe rather than a version string because that is the thing that actually
    has to be true, and because the failure this catches is a *partially* updated
    install, where the version number is already the new one.
    """
    from comfy.ldm.minimax.model import PackedLayout

    latent_h = latent_w = 4
    frame_rows = (latent_h // 2) * (latent_w // 2)
    guide = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 2, latent_h, latent_w)}
    layout = PackedLayout(1, 2, latent_h, latent_w, 1, keyframes=[guide])
    return int((~layout.img_update).sum()) == 2 * frame_rows


def assert_clip_guides_supported():
    """Refuse to pin a clip a core of this vintage will mis-size.

    Without this the render gets several minutes into the segment and then dies
    inside the DiT on a bare tensor shape, naming neither the guide nor ComfyUI.
    """
    global _CLIP_GUIDES
    if _CLIP_GUIDES is None:
        try:
            _CLIP_GUIDES = _probe_clip_guides()
        except Exception:
            _CLIP_GUIDES = True  # an unreadable core is not grounds for refusing to render
    if _CLIP_GUIDES:
        return
    raise RuntimeError(
        "This ComfyUI reserves one latent frame for a guide clip however long the "
        "clip is, so the handoff this segment opens on would write about twelve "
        "times the rows the sequence has room for -- the 'shape mismatch: value "
        "tensor of shape [A, 96] cannot be broadcast to indexing result of shape "
        "[B, 96]' that would come out of the sampler in a minute or two.\n\n"
        "Multi-frame guides arrived in ComfyUI together with MiniMaxH3AddGuide, in "
        "0.34.0. This install has that node -- the pack does not load without it -- "
        "but comfy/ldm/minimax/model.py is from before it, which is a half-applied "
        "update rather than an old one. Update ComfyUI, restart it fully, and check "
        "that comfy/ldm/minimax/model.py contains 'vt = video_latent.shape[2]'. If "
        "it does and this still fires, delete comfy/ldm/minimax/__pycache__.")


def generation_length(frames):
    """Snap a frame count UP onto H3's 17k+5 grid.

    H3 only generates 5, 22, 39, ... 124, 141, ... frames. The trained range is
    124-362 (about 5-15 s at 24 fps); shorter and longer both work but degrade.
    """
    frames = max(5, int(round(frames)))
    while frames % 17 != 5:
        frames += 1
    return frames


def guide_length(frames):
    """Snap a guide-clip frame count DOWN onto the same grid.

    ``MiniMaxH3AddGuide`` truncates guide clips to 17k+5 internally. Doing it here
    too means the handoff length we record in the manifest is the length actually
    anchored, so the join arithmetic and the anchor agree to the frame.
    """
    frames = int(round(frames))
    if frames < 5:
        return 0
    while frames % 17 != 5:
        frames -= 1
    return frames


def guide_length_up(frames):
    """Snap a guide-clip frame count UP onto the same grid.

    ``guide_length`` rounds *down* because it predicts what
    ``MiniMaxH3AddGuide`` does when we hand it more frames than it can keep.
    When we get to choose the request we want the opposite: the spec's 1.6 s
    handoff is 38 frames, which rounds *down* to 22 (0.92 s) — half the
    replay window quietly disappears. Rounding up to 39 (1.625 s) keeps the
    full 1.6 s and is a no-op for the truncation inside ``MiniMaxH3AddGuide``.
    """
    frames = int(round(frames))
    if frames <= 0:
        return 0
    if frames < 5:
        return 5
    while frames % 17 != 5:
        frames += 1
    return frames


def seconds_to_frames(seconds):
    return int(round(float(seconds) * FPS))


def audio_latent_frames(frames):
    """Audio latent steps covering ``frames`` pixel frames.

    H3 runs its audio latent at 40 Hz against 24 fps video. This is the same
    rounding ``_empty_av_latent`` uses to size a segment's audio stream, so a tail
    cut with it lines up with the tail of the stream it came from.
    """
    return int(round(float(frames) / FPS * AUDIO_LATENT_FPS))


def ordered_autogrow(values):
    """Autogrow inputs arrive as ``{"image_0": t, "image_3": t}``; return them in
    ordinal order, dropping unconnected slots but keeping the ordinal for pairing."""
    out = []
    for name, value in (values or {}).items():
        if value is None:
            continue
        tail = name.rsplit("_", 1)[-1]
        try:
            ordinal = int(tail)
        except ValueError:
            ordinal = 0
        out.append((ordinal, value))
    out.sort(key=lambda kv: kv[0])
    return out


def slice_audio(audio, start_seconds, duration_seconds):
    """Cut an AUDIO dict to a window, padding with silence if it falls short.

    The reference build did this with ffmpeg and hit the bug this function exists to
    make impossible: an anchor clip whose audio outran its video by six seconds.
    ``AddGuide`` crops audio against the *target's* remaining duration, never against
    the image clip beside it, so an over-long soundtrack is anchored in full and a
    segment with no dialogue of its own replays the previous segment's lines.
    """
    if audio is None:
        return None
    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    start = max(0, int(round(start_seconds * rate)))
    want = max(1, int(round(duration_seconds * rate)))
    cut = waveform[..., start:start + want].clone()
    if cut.shape[-1] < want:
        pad = torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype)
        cut = torch.cat([cut, pad], dim=-1)
    return {"waveform": cut, "sample_rate": rate}


def evict_dead_loaded_models() -> int:
    """Pop Comfy ``LoadedModel`` slots that ``free_memory`` would skip forever.

    ``is_dead()`` means the ModelPatcher weakref is gone while the shared
    MiniMaxH3 module is still alive (graph MODEL / Sage cycle). Those slots log
    ``Potential memory leak detected with model MiniMaxH3`` and then sit in
    ``current_loaded_models``, so later unloads cannot touch them.

    That is not cosmetic: each leftover slot keeps pinning a model's weights in
    RAM. This pack runs ~49 GB of weights against ~40 GB of physical memory, so
    one unreachable slot per segment is enough to push a long chain into paging
    (symptom: segment N's text encoding jumps from ~70 s to 5-12 minutes while
    VRAM sits idle). Evicting the slot does not copy weights -- it only restores
    unload bookkeeping.

    Lives here rather than in ``director`` so ``engine`` can call it too without
    importing ``director`` (which would be a cycle).
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return 0
    models = getattr(mm, "current_loaded_models", None)
    if not models:
        return 0
    evicted = 0
    for i in range(len(models) - 1, -1, -1):
        cur = models[i]
        try:
            if not cur.is_dead():
                continue
            models.pop(i)
            evicted += 1
        except Exception:
            continue
    return evicted
