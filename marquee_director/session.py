"""On-disk state for one chain.

A chain is not a single render, it is N renders that each take minutes. Keeping
them in a session folder is what makes the pack usable:

* **Resume.** Re-queueing after a crash, an OOM or a Ctrl-C reuses every segment
  whose settings did not change and only renders the rest.
* **Edit one segment.** Change segment 4's prompt and segments 1-3 are reused;
  4 onward re-render, because each cache key folds in the key of the segment
  before it.
* **Repair.** The repair node needs the previous segment's tail and the tail the
  *next* segment already continued from. Both are on disk.

布局（**全包只有 ``output_root()`` 这一处碰 ComfyUI 的输出目录**）::

    <ComfyUI 输出目录>/h3_continuous/<剧本名>/<版本名>/     现行
    <ComfyUI 输出目录>/h3_continuous/<会话名>/              升级前的单层布局

剧本名取 PACK 头部的 Project（或「H3 链路设置」上显式填的「剧本名称」），
版本名取「会话名」控件；版本名留空时自动 ``r1``/``r2``…，所以同一部剧本每次
**全新渲染**（复用关掉）都开一个新文件夹，上一版的产出原封不动还在。

★ 老的单层目录**原位沿用、不搬**。缓存键算的是内容而不是路径，但段文件躺在
  哪个目录是断点续渲的立足点：换了根，已渲的 18 段对新一轮而言不存在，
  等于凭空一次全量重渲。所以 ``resolve_session_dir`` 先认旧目录，只有旧目录
  没有产出（或本次明确要重开一轮）才落到两层布局。
"""

import hashlib
import json
import os
import re

import folder_paths

ROOT = "h3_continuous"
MANIFEST_VERSION = 2

# 剧本名与会话名都拿不到时的兜底目录。**全包只有这一个兜底常量** —— 以前
# Director 兜到 ``my_chain``、存档节点兜到 ``h3_pack``，同一份没带 session_name
# 的分镜 JSON 会写进两个不同目录，面板因此查不到自己刚存的档。
UNNAMED = "h3_pack"
# 自动版本号的词头：<剧本>/r1、<剧本>/r2 …
VERSION_PREFIX = "r"

# Windows 设备名。这些名字**即使带扩展名**也不能当目录名用：``os.makedirs``
# 不抛异常，但目录根本没被创建，之后每一次写段都变成 FileNotFoundError。
# 实测（2026-10-17 自检）：CON/COM1/aux/CON.txt 在本机反而能建出来，NUL 建不出来
# —— 正是这种"看运气"的差异最该在入口一刀切掉，而不是等某次渲染写到一半失败。
WIN_RESERVED = (
    {"CON", "PRN", "AUX", "NUL"}
    | {"COM%d" % i for i in range(1, 10)}
    | {"LPT%d" % i for i in range(1, 10)}
)


def sanitize(name):
    """Normalise a session name into a safe directory name.

    ★ **这是全包唯一的归一化实现**：会话名、剧本名、面板查找名、存档目录名
      全部走它（``common.safe_session_name`` 只是"按候选顺序挑第一个非空
      再交给本函数"的包装）。以前 common 里另抄了一份正则、两边靠 import 期
      的 monkey-patch 对齐，那是**导入顺序相关**的：任何只导入 ``session``
      而未触发 ``common`` 的路径都会拿到另一套规则，同一个名字算出两个目录 ——
      渲染写 ``v18_en_v10``、面板查 ``曹贼的性价比_v18_en_v10``，于是
      ``done=0``、断点续渲每次都从第 1 段重来。现在没有第二套可对齐。

    ★ 必须保留 CJK：会话名与剧本名常常就是中文片名
      （``曹贼的性价比_v18_en_v10``），保留它 = 磁盘上那个文件夹名就是人
      看得懂的片名。
    """
    # \w 在 Python 3 下是 Unicode 的，已包含 CJK；不用额外列 \u4e00-\u9fff。
    name = re.sub(r"[^\w.-]+", "_", (name or "").strip(), flags=re.UNICODE)
    name = re.sub(r"\.{2,}", "_", name)  # never let a session name walk up a directory
    name = name.strip("._") or UNNAMED
    name = name[:60]
    # Windows 设备名兜底：只看第一个点之前的部分（CON.txt 同样是设备名）。
    # 加下划线前缀而不是替换 —— 前缀后仍可读，且不会与别的会话撞名。
    if name.split(".", 1)[0].upper() in WIN_RESERVED:
        name = "_" + name
    return name


def output_root():
    """全包唯一的产出根。

    ComfyUI Desktop 的 ``--output-directory`` 是**可变的**（这台机器上它就换过
    几次），所以任何写死的输出根都会在某次改设置之后把产出分到两棵树里 ——
    表现就是老问题：面板说「会话目录不存在」、断点渲染从第 1 段重来。
    只有这一个函数碰 ``folder_paths``，其余模块（cleanup / json_io / director /
    routes）一律经由它。
    """
    return folder_paths.get_output_directory()


def store_root():
    """``<输出目录>/h3_continuous``，节点包全部产出的总根。"""
    return os.path.join(output_root(), ROOT)


# ``seg_01.mp4`` / ``seg_01.tail.wav`` / ``seg_01.frames.mkv`` … 都算段产物。
_SEG_FILE_RE = re.compile(r"^seg_\d+([._].+)?$")


def is_session_dir(path):
    """这个目录本身是一个会话的产出目录吗（而不是「剧本」那层分组夹）？

    判据只看产物，不看名字：有 manifest.json、有任一 ``seg_NN.*``、或者有和
    目录同名的成片，就算。剧本夹里只有 ``r1``/``r2`` 子目录，没有这些文件，
    于是不会被误当成会话 —— 一旦误认，清理扫描会把整部剧本的产出当"没成片
    的中间文件"扫掉。
    """
    if not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    here = os.path.basename(os.path.normpath(path))
    for name in names:
        if name == "manifest.json" or _SEG_FILE_RE.match(name):
            return True
        if name == here + ".mp4":
            return True
    return False


def iter_session_dirs(root=None):
    """列出总根下所有会话目录：新的两层（``<剧本>/<版本>``）与老的单层都认。

    深度刻意封在 2 —— 再深没有意义，而且会让清理扫描走进用户的其它目录。
    """
    root = root or store_root()
    out = []
    if not os.path.isdir(root):
        return out
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return out
    for entry in entries:
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        if is_session_dir(path):
            out.append(path)
            continue
        try:
            subs = sorted(os.listdir(path))
        except OSError:
            continue
        for sub in subs:
            leaf = os.path.join(path, sub)
            if os.path.isdir(leaf) and is_session_dir(leaf):
                out.append(leaf)
    return out


def script_of(session_dir, root=None):
    """会话目录归属的剧本夹名；老的单层布局没有这一层，返回 ''。"""
    parent = os.path.dirname(os.path.normpath(str(session_dir)))
    if os.path.normcase(parent) == os.path.normcase(root or store_root()):
        return ""
    return os.path.basename(parent)


def next_version(script_dir, fresh=False):
    """剧本目录下下一个该用的版本名。

    ``fresh``（复用关掉）才**另开一页**：``r1`` 渲完了，照同一部剧本再渲一版
    就落到 ``r2``，上一版一个字节都不动。``fresh=False`` 时接在最新那一版上
    —— 断点续渲要的就是这个：任何情况下都不能因为版本名换了而整条重渲。
    """
    highest = 0
    if os.path.isdir(script_dir):
        try:
            for sub in os.listdir(script_dir):
                match = re.match(r"^%s(\d+)$" % VERSION_PREFIX, sub)
                if match and os.path.isdir(os.path.join(script_dir, sub)):
                    highest = max(highest, int(match.group(1)))
        except OSError:
            pass
    if highest == 0:
        return "%s1" % VERSION_PREFIX
    return "%s%d" % (VERSION_PREFIX, highest + (1 if fresh else 0))


def resolve_session_dir(version="", script="", fresh=False):
    """``(会话名, 剧本名)`` -> ``(目录, 生效的剧本名, 生效的版本名)``。

    三条规矩，按优先级：

    1. **旧单层目录原位沿用**（除非本次明确重开一轮）。升级前
       ``h3_continuous/<会话名>/`` 是唯一布局，里面有已渲好的段；把它并进两层
       布局的话，本轮新段写进新目录、旧段留在旧目录，断点续渲立刻退化成整条
       重渲。所以旧目录只要有产出，就一直用旧目录。
    2. **知道剧本名** -> ``h3_continuous/<剧本名>/<版本名>/``。版本名留空（或跟
       剧本名撞成同一个词）就按 :func:`next_version` 自动取 ``r1``/``r2``…
    3. **只有会话名**（源视频切分、手搓 JSON，没处取剧本名）-> 维持单层
       ``h3_continuous/<会话名>/``。
    """
    v = sanitize(version) if str(version or "").strip() else ""
    s = sanitize(script) if str(script or "").strip() else ""
    store = store_root()

    if not fresh:
        for cand in (v, s):
            if not cand:
                continue
            legacy = os.path.join(store, cand)
            if is_session_dir(legacy):
                return legacy, "", cand

    if not s and not v:
        return os.path.join(store, UNNAMED), "", UNNAMED
    if not s:
        # 只有会话名（源视频切分、手搓 JSON，剧本名没处取）：维持单层，
        # 不硬造 ``<会话名>/<会话名>/`` 这种套娃目录。
        return os.path.join(store, v), "", v
    if not v or v == s:
        # 版本没填，或者跟剧本名撞成同一个词 —— 都交给自动版本号。
        v = next_version(os.path.join(store, s), fresh)
    return os.path.join(store, s, v), s, v


def digest(value):
    """Stable short digest of anything a segment input can be.

    Tensors are sampled rather than hashed whole: a 2048px reference image is
    50 MB and this runs for every segment on every queue. Shape, dtype and a
    strided slice change whenever the picture does, which is all a cache key needs.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        t = value
        step = max(1, t.shape[0] // 8) if t.ndim > 0 and t.shape[0] > 8 else 1
        sample = t[::step].contiguous().cpu().numpy().tobytes()
        h = hashlib.sha1()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(sample)
        return h.hexdigest()[:16]
    if isinstance(value, dict) and "waveform" in value:
        return "audio:%s:%s" % (digest(value["waveform"]), value.get("sample_rate"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(digest(v) for v in value) + "]"
    return hashlib.sha1(repr(value).encode("utf-8", "replace")).hexdigest()[:16]


def sampler_digest(sampler):
    """Stable fingerprint of a SAMPLER object.

    ``repr()`` will not do: it carries a memory address, so it changes on every
    queue and would invalidate the whole cache every run.
    """
    fn = getattr(sampler, "sampler_function", None)
    name = getattr(fn, "__name__", None) or type(sampler).__name__
    extra = getattr(sampler, "extra_options", None) or {}
    return hashlib.sha1((name + repr(sorted(extra.keys()))).encode()).hexdigest()[:16]


def model_digest(model):
    """Fingerprint the model patcher well enough to notice a LoRA change.

    Weights themselves are never hashed -- far too slow. This sees the model class
    and every patch key with its strengths, which covers adding, removing or
    re-weighting a LoRA. It does *not* see a swapped checkpoint file of the same
    class: rename the session or turn ``resume`` off when you change one.
    """
    try:
        parts = [type(getattr(model, "model", model)).__name__]
        for key, patches in sorted(getattr(model, "patches", {}).items()):
            strengths = []
            for patch in patches:
                strengths += [round(float(v), 6) for v in patch[:1] + patch[2:3]
                              if isinstance(v, (int, float))]
            parts.append("%s:%s" % (key, strengths))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    except Exception:  # never let a cache key break a render
        return "unknown"


# 渲染管线标识。改动 render_segment 的**渲染逻辑本身**（不是可调参数）时必须
# bump：它进缓存键，bump 会让所有段重渲 —— 这正是目的，旧缓存是旧逻辑的产物。
#
# 历史：
#   v1  cond 引导（_pin_latent 只写 minimax_keyframes）
#   v2  cond 引导 + latent 直写 + noise_mask（apply_latent_continue，2026-09-15）
#
# ⚠️ 已知混合产物：会话 v18_en_v5 的段 1/2 是 v1 渲染的、段 3/4 是 v2 渲染的。
#    尾巴 latent 与渲染路径无关所以成片仍能接上，但不算干净。想让整条链一致，
#    把这个值改成 "h3_render_v2" 即可强制段 1..N 全部重渲（约 8 分钟/段）。
#    平时保持 None：既有会话的 payload 保持字节一致，不会因升级而无故重渲。
RENDER_PIPELINE_ID = None

# seam_redraw 的默认值，必须与 engine.SEAM_MIN_MASK 一致。
# session.py 不 import engine（避免循环依赖），所以在这里再写一遍。
SEAM_REDRAW_DEFAULT = 0.10


def segment_key(settings, segment, handoff, previous_key):
    """Fingerprint of everything that changes what this segment renders.

    ``previous_key`` is in the hash on purpose: a segment is a continuation of the
    one before it, so editing segment 2 must invalidate 3..N even though their own
    prompts are untouched.
    """
    payload = {
        "prev": previous_key,
        "prompt": segment["prompt"].strip(),
        "length": segment["length"],
        "handoff": handoff,
        "seed": segment["resolved_seed"],
        "width": settings["width"],
        "height": settings["height"],
        "ref_image_size": settings["ref_image_size"],
        "sigmas": digest(settings["sigmas"]),
        "sampler": sampler_digest(settings["sampler"]),
        "model": model_digest(settings["model"]),
        "images": [digest(t) for t in segment["images"]],
        "videos": [digest(t) for _, t in segment["videos"]],
        "video_audios": [digest(a) for _, a in segment["video_audios"]],
        "audios": [digest(a) for a in segment["audios"]],
    }
    # Only a segment that actually consumes an anchor cares how that anchor was
    # built, and the key is left byte-identical when it does not. That keeps the
    # first segment of every pre-existing session cached across this upgrade -- the
    # segment renders the same either way, and its own outgoing tail is now written
    # in both forms regardless. Segments 2..N do change, because their handoff
    # genuinely did.
    if previous_key and settings.get("handoff_mode"):
        payload["handoff_mode"] = settings["handoff_mode"]
    if previous_key and settings.get("drift_arrest"):
        payload["drift_arrest"] = round(float(settings["drift_arrest"]), 4)
    # 逐 step 重绘掩码（seam_remask）：与上面两项同一条规则 —— 只有**真的开了**
    # 且这一段确实消费锚点时才并入。默认关，所以升级后既有会话一段都不用重渲；
    # 打开后只有段 2..N 重渲，首段没有锚点、不受影响，缓存照旧命中。
    if previous_key and settings.get("seam_remask"):
        payload["seam_remask"] = True
    # 渲染方式参数：只在**显式设置**时才并入。既有会话的 settings 没有这两个键，
    # payload 就保持字节一致、不会因升级而无故重渲；一旦真设了，改动才会正确
    # 触发重渲 —— 否则改了渲染方式缓存却认为没变，会拿到旧路径渲染的结果。
    if "handoff_remask" in settings:
        payload["handoff_remask"] = bool(settings["handoff_remask"])
    # seam_redraw 默认 0.10（== engine.SEAM_MIN_MASK）。**等于默认值时不并入
    # payload**：H3ChainSettings 现在总是传出这个键，无条件并入会让所有既有
    # 会话的段 2..N 全部重渲。只有真的调过才进键、才触发重渲。
    seam = settings.get("seam_redraw")
    if seam is None:
        seam = settings.get("seam_min_mask")          # 旧名兼容
    if seam is not None and abs(float(seam) - SEAM_REDRAW_DEFAULT) > 1e-6:
        payload["seam_redraw"] = round(float(seam), 4)
    # TeaCache 阈值（H3 链路设置）：**等于 0（默认=关）时不并入** ——
    # H3ChainSettings 现在总是传出这个键，无条件并入会让所有既有会话的段
    # 全量重渲。开了之后改阈值会正确触发重渲：它改变成片本身（被跳过的
    # 步输出的是复用值，不是真实 forward 的结果）。
    tc = settings.get("teacache_thresh")
    if tc is not None and float(tc) > 0.0:
        payload["teacache_thresh"] = round(float(tc), 4)
    if RENDER_PIPELINE_ID:
        payload["pipe"] = RENDER_PIPELINE_ID
    # 二采精修（H3 二采精修）：同样只在**接了 refine 且给了噪声表**时才并入。
    # 没接时 payload 保持字节一致 —— 既有会话不会因为多出这个功能而全量重渲。
    # 接了之后改任何一个二采参数都会正确触发重渲（二采改的是成片本身）。
    refine = settings.get("refine")
    if isinstance(refine, dict) and refine.get("sigmas") is not None:
        payload["refine"] = {
            "sigmas": digest(refine.get("sigmas")),
            # 不接 refine_model 时二采用的是主模型，其摘要已经在上面对过；
            # 这里再记一次只是为了「接/不接二采模型」这两种情形能区分开。
            "model": model_digest(refine.get("model") or settings["model"]),
            "passes": int(refine.get("passes") or 1),
            # 二采采样器存的是名字（字符串），不是 sampler 对象：对象每次运行
            # 新建、repr 带内存地址，直接进键会让缓存永远不命中。
            "sampler": refine.get("sampler") or 0,
            "seed_mode": refine.get("seed_mode"),
            "seed": int(refine.get("seed") or 0),
        }
    # 脸部修复（H3 脸部修复）：与 refine 同一条规则 —— 只在接了且给了噪声表时
    # 才并入。它改的是本段解码后的像素（贴回的人脸特写），段间锚点仍来自未修的
    # latent，所以**不会**连带影响下一段；改修脸参数只需要本段重渲。
    # 注意 detector / confidence / crop_factor 这些也必须进键：换检测器框出来的
    # 脸不一样，成片就不一样，缓存不能认为没变。
    face = settings.get("face_refine")
    if isinstance(face, dict) and face.get("sigmas") is not None:
        payload["face"] = {
            "sigmas": digest(face.get("sigmas")),
            "detector": str(face.get("detector") or ""),
            "confidence": round(float(face.get("confidence") or 0.0), 4),
            "select": face.get("select"),
            "crop_factor": round(float(face.get("crop_factor") or 0.0), 4),
            "canvas_mode": face.get("canvas_mode"),
            "canvas": [int(face.get("canvas_width") or 0),
                       int(face.get("canvas_height") or 0)],
            "sampler": face.get("sampler") or 0,
            "seed_mode": face.get("seed_mode"),
            "paste_region": face.get("paste_region"),
            "feather": int(face.get("feather") or 0),
            "blend": round(float(face.get("blend") or 0.0), 4),
            "seam_fade": face.get("seam_fade"),
            # ★ 质量闭环 + 强度插值十键（2026-10-14 补收）。face_refine 的节点
            #   execute 一直都下发它们，注释也写着"全部进 pack、pack 参与缓存键"，
            #   但这里只收了上面 14 个 —— 漏收的后果不是报错，而是**静默复用缓存**：
            #   resume 开着调修脸强度 / 质检阈值，缓存键不变，段被判"未变"直接
            #   复用旧段，成片与新设置不符且零提示。这些键全部直接改变本段像素
            #   （强度决定贴回多柔，identity/detail/stability 判据决定多版尝试里
            #   保留哪一版），必须进键。
            #   注意键只在"接了修脸且给了噪声表"时存在 —— 没接的面无此键，
            #   既有会话的 payload 保持字节一致，不会因此全量重渲。
            "quality_check": bool(face.get("quality_check")),
            "max_attempts": int(face.get("max_attempts") or 0),
            "identity_floor": round(float(face.get("identity_floor") or 0.0), 4),
            "detail_floor": round(float(face.get("detail_floor") or 0.0), 4),
            "stability_floor": round(float(face.get("stability_floor") or 0.0), 4),
            "adaptive_denoise": bool(face.get("adaptive_denoise")),
            "strength_small_face": round(float(face.get("strength_small_face") or 0.0), 4),
            "strength_large_face": round(float(face.get("strength_large_face") or 0.0), 4),
            "face_px_small": int(face.get("face_px_small") or 0),
            "face_px_large": int(face.get("face_px_large") or 0),
        }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def signature_to_record(signature):
    """A latent signature as plain JSON -- 24 means and 24 spreads."""
    if not signature:
        return None
    return {"mean": [round(float(v), 6) for v in signature["mean"]],
            "std": [round(float(v), 6) for v in signature["std"]]}


def signature_from_record(record):
    """The inverse, for a segment being reused from cache.

    Returns None for a session rendered before signatures existed, which simply
    means drift arrest has no reference to work from until something re-renders.
    """
    import torch

    payload = (record or {}).get("signature")
    if not payload:
        return None
    return {"mean": torch.tensor(payload["mean"]), "std": torch.tensor(payload["std"])}


class Session:
    """一个会话的磁盘目录。

    ``Session(会话名)`` 与升级前完全等价（单层）。传了 ``script`` 才走
    ``<剧本>/<版本>`` 两层，并且只在目标目录还没有产出时才生效 —— 见
    :func:`resolve_session_dir` 第 1 条规矩。
    """

    def __init__(self, name, script=None, fresh=False):
        self.dir, self.script, version = resolve_session_dir(
            name, script=script or "", fresh=fresh)
        # 目录名就是版本名，成片名跟着目录走（``<版本>.mp4``）。以前成片名取
        # ``sanitize(name)``，一旦目录被旧布局截胡，两者就会分叉 —— 面板和清理
        # 都是按"目录同名 mp4"认成片的，那里用 basename、这里用 name，
        # 成片就永远认不出来。现在两边由同一个表达式推出。
        self.name = version
        os.makedirs(self.dir, exist_ok=True)
        # 断点续渲的"链条完整性"开关。见 ``note_render`` / ``cached``：
        # 磁盘补齐的记录是靠"文件还在"判定命中的，一旦本次运行真的重渲了
        # 某一段，它后面所有补齐的段都不能再算命中 —— 否则会出现"第 3 段
        # 是新渲的、第 4 段却接着老的第 3 段尾巴"这种接不上的片子。
        self.resume_intact = True

    @classmethod
    def from_dir(cls, path):
        """直接认领一个**已经算好的**会话目录（chain 里带着它走）。

        两层布局以后，光凭"会话名"三个字推不出目录 —— 还差剧本名那一层。
        chain 里既然已经带了 ``dir``，就照它来，不再重新解析一遍：重算的每一次
        机会都是一次分叉（渲染写 ``<剧本>/r2``、读档去 ``<剧本>/r1``）。
        """
        self = cls.__new__(cls)
        self.dir = os.path.normpath(str(path))
        self.name = os.path.basename(self.dir)
        self.script = script_of(self.dir)
        self.resume_intact = True
        return self

    @classmethod
    def from_settings(cls, settings, fresh=False):
        """按 settings 里 carries 的 (会话名, 剧本名) 打开会话。

        ★ **所有构造点都必须走这里**：Director 主循环、修复分支、渲染节点、
          时间线行都得到同一个目录。以前五处各自 ``Session(settings["session_name"])``，
          剧本名一旦参与进来，漏传的那处就会算出另一个目录，
          修好的段和渲染的段分居两地。
        """
        return cls(str((settings or {}).get("session_name") or ""),
                   script=(settings or {}).get("script_name"), fresh=fresh)

    # --- paths -----------------------------------------------------------------
    def segment_path(self, index):
        return os.path.join(self.dir, "seg_%02d.mp4" % (index + 1))

    def tail_path(self, index):
        """The clip segment index+1 opens on. Kept even when a segment is repaired:
        it is the thing downstream already continues from."""
        return os.path.join(self.dir, "seg_%02d.tail.mp4" % (index + 1))

    def joined_path(self):
        return os.path.join(self.dir, "%s.mp4" % self.name)

    @property
    def manifest_path(self):
        return os.path.join(self.dir, "manifest.json")

    # --- manifest --------------------------------------------------------------
    def load(self):
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("version") != MANIFEST_VERSION:
            return None
        return data

    def save(self, records, extra=None):
        data = {"version": MANIFEST_VERSION, "session": self.name,
                "segments": records}
        if self.script:
            # 只记剧本名，**不记绝对路径**：整棵 output 树是可以被用户搬走/改
            # --output-directory 的，manifest 里一旦出现绝对路径就变成第二次
            # "目录分叉"的源头。
            data["script"] = self.script
        data.update(extra or {})
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.manifest_path)
        return data

    def disk_segments(self, limit=9999):
        """How many segments are already on disk, counting up from the first.

        Only a *contiguous* run counts: a session with 1, 2, 3 and 5 is four
        segments of progress, not five, because 4 is missing and 5 no longer
        continues from anything.
        """
        count = 0
        while count < limit and os.path.exists(self.segment_path(count)):
            count += 1
        return count

    def disk_segment_indices(self):
        """1-based indices that actually have a ``seg_NN.mp4`` on disk.

        Unlike :meth:`disk_segments` this does not stop at the first gap, so an
        orphaned segment (say 5 when 4 is missing) is still reported. That file
        is real output: the timeline shows its preview, so the "rendered" count
        must not silently exclude it and then disagree with the previews.
        Resuming still uses :meth:`disk_segments`, which is the contiguous run.
        """
        out = []
        try:
            for fn in os.listdir(self.dir):
                m = re.match(r"^seg_(\d+)\.mp4$", fn)
                if m:
                    out.append(int(m.group(1)))
        except OSError:
            return []
        return sorted(set(out))

    def probe_tail_frames(self, index):
        """Best-effort frame count of segment ``index``'s handoff clip.

        Only used to fill in the ``handoff`` of a record being recovered from
        disk. Without it the join does not know how many replayed frames to drop
        and every cut stutters. Imported lazily: this module is otherwise
        importable without torch, which keeps it testable on its own.
        """
        try:
            from . import video_io
            return int(video_io.frame_count(self.tail_path(index)))
        except Exception:      # never let a probe break a resume
            return 0

    def probe_segment_frames(self, index):
        """Best-effort frame count of segment ``index``'s own render.

        Needed by ``reconcile()``: a record rebuilt from disk used to carry a
        hard-coded ``length=0``, and ``director`` sums ``length`` to size the
        join (``total = sum(length) - sum(handoff[:-1])``). With every recovered
        record contributing 0, a session restored after its manifest was swept
        joined to <= 0 frames. Probe the real file instead -- it is the same
        lazy-import / never-raise contract as ``probe_tail_frames``.
        """
        try:
            from . import video_io
            return int(video_io.frame_count(self.segment_path(index)))
        except Exception:      # never let a probe break a resume
            return 0

    def reconcile(self, manifest, limit=9999):
        """补齐 manifest 里缺失的段记录：磁盘上有 seg_NN.mp4，manifest 却没记。

        怎么会发生：段是每段跑完就落盘的，manifest 也每段写一次，但一次
        「只渲 1 段」的排队（或画完之后被截断的 shots_json）会把
        ``records[:index] + [record]`` 写成只有一条，而 seg_02..seg_04 早就
        躺在磁盘上了。此时重跑，2..4 会被当成"没渲过"重新烧一遍显卡。

        补齐的记录打 ``recovered`` 标记。原始缓存键已经无从复原（它是
        prompt/参考图/上一段 key 的哈希），所以只要文件还在就认为这一段
        已完成，见 ``cached``。本次运行命中时会带着真实 key 重新写回
        manifest，下一次排队就回到普通的键比对。
        """
        on_disk = self.disk_segments(limit)
        # 既没有 manifest 又一段都没渲：没什么可补的，返回 None，别伪造一个空
        # manifest 出来 —— 调用方（比如修复节点）靠 None 判断"会话还没跑过"。
        if not manifest and not on_disk:
            return None
        if not manifest:
            manifest = {"version": MANIFEST_VERSION, "session": self.name,
                        "segments": [], "recovered": 0}
        segments = list(manifest.get("segments") or [])
        known = set()
        for i, rec in enumerate(segments):
            if isinstance(rec, dict):
                known.add(int(rec.get("index", i)))
        # 录的是帧数（和真实记录一致）。拼接靠它跳过回放帧，写 0 会让
        # 每个接口都多出一段重复的开头；而 director 又拿 sum(length) 定拼接
        # 总长，写 0 会让总长算成 0。所以这里必须真的去探一下文件。
        try:
            _fps = float(manifest.get("fps") or 0) or 0.0
        except (TypeError, ValueError):
            _fps = 0.0
        added = 0
        _burned_dropped = 0
        for i in range(on_disk):
            if i in known:
                continue
            # ★ 2026-10-02：采纳无记录的磁盘段前，先 OCR 扫一遍底部字幕带。
            #   烧字的段（典型：字幕质检重渲途中队列被取消，烧字段留在盘上、
            #   manifest 记录又丢失）一旦被采纳就会静默拼进成片 —— v25 实测。
            #   命中即删段文件（连同尾锚），让本段走正常重渲。宁可重烧，不可带伤交付。
            try:
                from .subtitle_qc import scan as _qc_scan
                _qc_hits = _qc_scan(self.segment_path(i))
            except Exception:
                _qc_hits = []
            if _qc_hits:
                _burned_dropped += 1
                try:
                    os.remove(self.segment_path(i))
                    _tp = self.tail_path(i)
                    if os.path.exists(_tp):
                        os.remove(_tp)
                except OSError:
                    pass
                from .common import warn  # 局部 import：common 顶部就 import session，模块级会成环
                warn("reconcile: seg_%02d 采纳前 OCR 命中烧字 %s"
                     " —— 已删除，本段将重渲", i + 1, _qc_hits)
                continue
            has_tail = os.path.exists(self.tail_path(i))
            _len = self.probe_segment_frames(i)
            segments.append({
                "index": i,
                "key": None,
                "file": os.path.basename(self.segment_path(i)),
                "length": _len,
                "handoff": self.probe_tail_frames(i) if has_tail else 0,
                "seconds": round(_len / _fps, 3) if (_len and _fps) else 0.0,
                "seed": 0,
                "prompt": "", "signature": None,
                "has_tail": has_tail,
                "recovered": True,
            })
            added += 1
        if added:
            segments.sort(key=lambda r: int(r.get("index", 0)) if isinstance(r, dict) else 0)
            manifest = dict(manifest)
            manifest["segments"] = segments
            manifest["recovered"] = int(manifest.get("recovered") or 0) + added
        return manifest

    def note_render(self):
        """这一段是真的重渲了 —— 后面所有"磁盘补齐"的段都不再可信。"""
        self.resume_intact = False

    def cached(self, manifest, index, key, needs_tail):
        """True when segment ``index`` on disk is still exactly what we would render."""
        if not manifest:
            return False
        segments = manifest.get("segments", [])
        if index >= len(segments):
            return False
        record = segments[index]
        if not os.path.exists(self.segment_path(index)):
            return False
        if needs_tail and not os.path.exists(self.tail_path(index)):
            return False
        # 磁盘补齐的记录（见 ``reconcile``）：键无从复原，文件还在就算完成。
        # 但前提是本次运行到这儿为止一段都没重渲过，否则链条已经断了。
        if record.get("recovered"):
            return bool(self.resume_intact)
        if record.get("key") != key:
            return False
        return True
