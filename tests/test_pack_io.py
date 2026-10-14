# -*- coding: utf-8 -*-
"""2026-10-14 回归：导演包 zip 导入/导出（借鉴 ComfyUI_MiniMaxH3_Director
pack.py 的安全骨架，格式为本包自有 .hpack.zip）。

守六条：
  ① roundtrip：build → import，storyboard / script / 参考图三者原样回来，
     槽位号 = <Picture N>，storyboard.json 落到会话目录；
  ② 路径穿越（``../x.png``、绝对路径、``refs/../../x``）：拒绝；
  ③ 压缩炸弹与单文件超限：拒绝（上限被打到小值用构造 zip 验证）；
  ④ 非本包格式 / 缺 pack.json 的 zip：拒绝；
  ⑤ 扩展名白名单外的成员（.exe/.sh/.py）：拒绝；
  ⑥ build 侧：槽位 1-9 之外的键报错、input 里找不到文件报错。

跑法（ComfyUI venv）：
    <ComfyUI>/.venv/Scripts/python.exe tests/test_pack_io.py
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY_ROOT = os.path.dirname(os.path.dirname(PKG))
sys.path.insert(0, COMFY_ROOT)
sys.path.insert(0, os.path.dirname(PKG))
sys.path.insert(0, PKG)

import folder_paths                                                  # noqa: E402

from marquee_director import pack as PK                              # noqa: E402

PNG_1X1 = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
           b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
           b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82")

STORYBOARD = json.dumps({
    "session_name": "r3", "script_name": "测试剧",
    "shots": [{"index": 1, "prompt": "a wide of a street", "seconds": 8}],
}, ensure_ascii=False)
SCRIPT = "[Shot 1] At 00:00.000 a wide of a street. -"


def _input_root(patch_target):
    tmp = tempfile.mkdtemp(prefix="h3_pack_input_")
    return Path(tmp), mock.patch.object(folder_paths, "get_input_directory",
                                        return_value=tmp)


def test_roundtrip():
    input_dir, patch = _input_root(None)
    patch.start()
    try:
        (input_dir / "kate.png").write_bytes(PNG_1X1)
        (input_dir / "stage.png").write_bytes(PNG_1X1)
        zip_path = PK.build_pack(
            meta={"script_name": "测试剧", "session_name": "r3", "shots": 1},
            storyboard=STORYBOARD, script=SCRIPT,
            refs={1: "kate.png", 2: "stage.png"})
        assert zip_path.is_file() and zip_path.name.endswith(PK.PACK_SUFFIX)

        result = PK.import_pack(zip_path)
        assert result["script"] == SCRIPT
        assert json.loads(result["storyboard"])["script_name"] == "测试剧"
        slots = {s["slot"]: s["path"] for s in result["slots"]}
        assert slots == {1: "h3_director_packs/测试剧/Picture1.png",
                         2: "h3_director_packs/测试剧/Picture2.png"}, slots
        dest = Path(folder_paths.get_input_directory()) / slots[1]
        assert dest.is_file() and dest.read_bytes() == PNG_1X1
        # storyboard.json 落进了会话目录（两层布局：<剧本>/<版本>/）
        assert result["storyboard_path"].endswith("storyboard.json")
        assert os.path.isfile(result["storyboard_path"]), result["storyboard_path"]
        print("PASS ① roundtrip：三方数据原样 + storyboard 落会话目录")
    finally:
        patch.stop()


def _evil_zip(entries):
    """entries: [(name, bytes_or_str)] → bytes"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries:
            zf.writestr(name, data)
    return buf.getvalue()


def _import_bytes(data):
    tmp = tempfile.mkdtemp(prefix="h3_pack_zip_")
    p = Path(tmp) / "evil.zip"
    p.write_bytes(data)
    try:
        return PK.import_pack(p)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


_BASE = [("pack.json", json.dumps({"format": PK.PACK_FORMAT, "version": 1})),
         ("storyboard.json", STORYBOARD)]


def test_path_traversal_rejected():
    for name in ("../evil.png", "/abs/evil.png", "refs/../../evil.png",
                 "refs\\..\\..\\evil.png", "refs/Picture<script>.png"):
        data = _evil_zip(_BASE + [(name, PNG_1X1)])
        try:
            _import_bytes(data)
            raise AssertionError("放行了恶意路径: %r" % name)
        except PK.PackError:
            pass
    print("PASS ② 路径穿越 / 绝对路径 / 非 ASCII 全部拒绝")


def test_limits_rejected():
    real = (PK.MAX_ZIP_ENTRIES, PK.MAX_UNCOMPRESSED, PK.MAX_SINGLE_FILE)
    try:
        # 单文件超限：条目数上限保持默认（够用），只压单文件上限到 1 MB
        PK.MAX_SINGLE_FILE = 1 << 20
        data = _evil_zip(_BASE + [("refs/Picture1.png", b"\0" * (2 << 20))])
        try:
            _import_bytes(data)
            raise AssertionError("放行了超大单文件")
        except PK.PackError as exc:
            assert "单文件" in str(exc)

        # 条目数超限
        PK.MAX_SINGLE_FILE = 1 << 25
        PK.MAX_ZIP_ENTRIES = 3
        data = _evil_zip(_BASE + [("refs/Picture1.png", PNG_1X1),
                                  ("refs/Picture2.png", PNG_1X1)])
        try:
            _import_bytes(data)
            raise AssertionError("放行了超条目包")
        except PK.PackError as exc:
            assert "条目" in str(exc)

        # 解压总量超限：条目数放开，总量压到 2 MB，两个 1 MB 成员 + 文本成员
        PK.MAX_ZIP_ENTRIES = 100
        PK.MAX_UNCOMPRESSED = 2 << 20
        data = _evil_zip(_BASE + [("refs/Picture1.png", b"\0" * (1 << 20)),
                                  ("refs/Picture2.png", b"\0" * (1 << 20))])
        try:
            _import_bytes(data)
            raise AssertionError("放行了超容量包")
        except PK.PackError as exc:
            assert "容量" in str(exc)
    finally:
        PK.MAX_ZIP_ENTRIES, PK.MAX_UNCOMPRESSED, PK.MAX_SINGLE_FILE = real
    print("PASS ③ 条目数 / 单文件 / 解压总量上限拦截")


def test_foreign_zip_rejected():
    # 不是本包格式
    data = _evil_zip([("pack.json", json.dumps({"format": "other-pack"})),
                      ("storyboard.json", STORYBOARD)])
    try:
        _import_bytes(data)
        raise AssertionError("放行了外格式包")
    except PK.PackError as exc:
        assert "格式" in str(exc), exc
    # 缺 pack.json
    data = _evil_zip([("storyboard.json", STORYBOARD)])
    try:
        _import_bytes(data)
        raise AssertionError("放行了裸 zip")
    except PK.PackError:
        pass
    # 缺 storyboard.json
    data = _evil_zip([("pack.json", json.dumps({"format": PK.PACK_FORMAT}))])
    try:
        _import_bytes(data)
        raise AssertionError("放行了无分镜包")
    except PK.PackError:
        pass
    print("PASS ④ 外格式 / 缺 pack.json / 缺 storyboard 全部拒绝")


def test_extension_whitelist():
    data = _evil_zip(_BASE + [("refs/Picture1.sh", "rm -rf /"),
                              ("refs/Picture2.png", PNG_1X1)])
    try:
        _import_bytes(data)
        raise AssertionError("放行了 .sh")
    except PK.PackError as exc:
        assert "不允许" in str(exc)
    print("PASS ⑤ 扩展名白名单")


def test_build_side_guards():
    input_dir, patch = _input_root(None)
    patch.start()
    try:
        # 槽位越界
        try:
            PK.build_pack(meta={}, storyboard=STORYBOARD,
                          refs={10: "kate.png"})
            raise AssertionError("放行了槽位 10")
        except PK.PackError:
            pass
        # input 里找不到文件
        try:
            PK.build_pack(meta={}, storyboard=STORYBOARD,
                          refs={1: "missing.png"})
            raise AssertionError("放行了不存在的参考图")
        except PK.PackError:
            pass
    finally:
        patch.stop()
    print("PASS ⑥ build 侧：槽位 1-9 / 文件必须存在")


def test_storyboard_from_pack_text():
    """⑦ storyboard 由服务端从 PACK 原文生成（与存档节点同源的
    to_shots_json）——包里那份必须能被读档节点 / 导演直接吃。"""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample = os.path.join(repo, "tools", "pack_sample.txt")
    if not os.path.isfile(sample):
        print("SKIP ⑦ 缺 tools/pack_sample.txt"); return
    text = open(sample, encoding="utf-8").read()
    storyboard, script, name = PK.storyboard_from_pack_text(text, "")
    data = json.loads(storyboard)
    assert script == "曹贼的性价比", script
    assert len(data.get("shots_info") or []) == 4
    first = data["shots_info"][0]
    for key in ("id", "duration", "gen_seconds", "ref_images", "task"):
        assert key in first, key
    # 空文本必须报错而不是产出空包
    try:
        PK.storyboard_from_pack_text("", "")
        raise AssertionError("空 PACK 放行了")
    except PK.PackError:
        pass
    print("PASS ⑦ storyboard 服务端生成：真实 PACK → 4 段 shots_info")


if __name__ == "__main__":
    test_roundtrip()
    test_path_traversal_rejected()
    test_limits_rejected()
    test_foreign_zip_rejected()
    test_extension_whitelist()
    test_build_side_guards()
    test_storyboard_from_pack_text()
    print("\n全部通过：导演包导入/导出的安全与数据完整性守住了。")
