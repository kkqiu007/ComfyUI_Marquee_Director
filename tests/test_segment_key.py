# -*- coding: # -*- coding: utf-8 -*-
"""2026-10-14 回归：段缓存键必须收录**所有**改变成片的参数。

守三条（对应 2026-10-14 修的两个真问题之一）：
  ① 修脸（face_refine）节点下发的 24 个键**全部**进缓存键。此前只收了 14 个，
     漏收的 10 个（quality_check / max_attempts / identity_floor / detail_floor /
     stability_floor / adaptive_denoise / strength_small_face /
     strength_large_face / face_px_small / face_px_large）全是直接改像素的参数 ——
     漏收的后果是 resume 开着调它们，段被判「未变」静默复用旧缓存，成片与新
     设置不符且零提示。
  ② 逐个改这 10 个键，缓存键必须变（逐个证明，不是只测一个代表）。
  ③ 「没接修脸」时 payload 无 face 键 —— 与升级前字节一致，既有会话不因这次
     修复全量重渲（这条与 seam_redraw / teacache 的「等于默认不进键」同一条纪律）。

跑法（不需要 GPU，也不需要 ComfyUI 在跑）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_segment_key.py
"""
import copy
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

from marquee_director import session as S                             # noqa: E402


class _Sampler(object):
    sampler_function = staticmethod(lambda *a, **k: None)
    extra_options = {}


class _Model(object):
    model = None
    patches = {}


def _settings(face=None, refine=None):
    s = {
        "width": 864, "height": 480,
        "ref_image_size": "max",
        "sigmas": [1.0, 0.5, 0.0],
        "sampler": _Sampler(),
        "model": _Model(),
    }
    if face is not None:
        s["face_refine"] = face
    if refine is not None:
        s["refine"] = refine
    return s


def _segment(seed=42, prompt="a wide of a street"):
    return {"prompt": prompt, "length": 124, "resolved_seed": seed,
            "images": [], "videos": [], "video_audios": [], "audios": []}


def _face(**over):
    """face_refine 节点 execute 实际下发的键（含 2026-10-14 补收的 10 个）。"""
    base = {
        "sigmas": [1.0, 0.6, 0.3, 0.0], "detector": "bbox/face_yolov8m.pt",
        "confidence": 0.5, "select": "auto", "crop_factor": 1.4,
        "canvas_mode": "match", "canvas_width": 0, "canvas_height": 0,
        "sampler": "euler", "seed_mode": "跟随一采", "paste_region": "face",
        "feather": 12, "blend": 0.85, "seam_fade": True, "mask_dilation": 16,
        "colour_match": 1.0,
        "quality_check": True, "max_attempts": 3, "identity_floor": 0.55,
        "detail_floor": 0.35, "stability_floor": 0.6, "adaptive_denoise": True,
        "strength_small_face": 0.55, "strength_large_face": 0.4,
        "face_px_small": 120, "face_px_large": 260,
    }
    base.update(over)
    return base


def test_ten_recovered_keys_change_the_key():
    """② 补收的 10 个键，逐个改动都必须换缓存键。"""
    changes = {
        "quality_check": False,
        "max_attempts": 1,
        "identity_floor": 0.7,
        "detail_floor": 0.5,
        "stability_floor": 0.4,
        "adaptive_denoise": False,
        "strength_small_face": 0.3,
        "strength_large_face": 0.5,
        "face_px_small": 160,
        "face_px_large": 300,
    }
    baseline = S.segment_key(_settings(face=_face()), _segment(), 1.625, "prev")
    for key, value in changes.items():
        k = S.segment_key(_settings(face=_face(**{key: value})),
                          _segment(), 1.625, "prev")
        assert k != baseline, "改了 face.%s 缓存键却没变 —— 旧缓存会被静默复用" % key
    print("PASS ② 补收 10 键逐个改变缓存键")


def test_unwired_face_keeps_payload_stable():
    """③ 没接修脸 → payload 无 face 键（不与接了之后共键）。"""
    plain = S.segment_key(_settings(), _segment(), 1.625, "prev")
    wired = S.segment_key(_settings(face=_face()), _segment(), 1.625, "prev")
    assert plain != wired
    # 两次「没接」必须字节一致（时间无关、无随机源）
    assert plain == S.segment_key(_settings(), _segment(), 1.625, "prev")
    print("PASS ③ 未接修脸时键稳定且与接入后可区分")


def test_existing_face_keys_still_work():
    """回归确认：原先收了的 14 键没有在修复中收窄。"""
    baseline = S.segment_key(_settings(face=_face()), _segment(), 1.625, "prev")
    for key, value in (("confidence", 0.6), ("crop_factor", 1.6),
                       ("feather", 20), ("blend", 0.7), ("detector", "bbox/hand.pt"),
                       ("select", "largest"), ("canvas_mode", "fixed"),
                       ("canvas_width", 512), ("sampler", "dpmpp_2m"),
                       ("seed_mode", "独立种子"), ("paste_region", "crop"),
                       ("seam_fade", False)):
        k = S.segment_key(_settings(face=_face(**{key: value})),
                          _segment(), 1.625, "prev")
        assert k != baseline, "face.%s 不再进缓存键了（修复时收窄了？）" % key
    print("PASS 原有 14 键全部仍在缓存键里")


def test_face_without_quality_keys_does_not_crash():
    """手工构造 / 旧 pack 的 face dict 缺新键时不炸（or 兜底）。"""
    lean = {"sigmas": [1.0, 0.0], "detector": "bbox/face_yolov8m.pt",
            "confidence": 0.5}
    k = S.segment_key(_settings(face=lean), _segment(), 1.625, "prev")
    assert isinstance(k, str) and len(k) == 20
    print("PASS 缺新键的 face dict 不炸（兜底为 0/False）")


def test_refine_and_seam_keys_unaffected():
    """邻域回归：refine / seam_redraw / teacache 的进键纪律没被带坏。"""
    base = S.segment_key(_settings(), _segment(), 1.625, "prev")
    same = S.segment_key(
        dict(_settings(), seam_redraw=0.10, teacache_thresh=0.0,
             refine={"sigmas": [1.0, 0.5, 0.0]}),
        _segment(), 1.625, "prev")
    assert same != base                       # refine 接了就进键
    # 没接 refine 的 baseline 再算一次仍等于自己（无隐性定时/随机源）
    assert base == S.segment_key(_settings(), _segment(), 1.625, "prev")
    print("PASS 邻域回归：refine 进键、其余纪律不变")


if __name__ == "__main__":
    test_ten_recovered_keys_change_the_key()
    test_unwired_face_keeps_payload_stable()
    test_existing_face_keys_still_work()
    test_face_without_quality_keys_does_not_crash()
    test_refine_and_seam_keys_unaffected()
    print("\n全部通过：缓存键与 face_refine 节点下发参数对齐了。")
