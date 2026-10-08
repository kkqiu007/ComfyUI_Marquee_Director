# -*- coding: utf-8 -*-
"""2026-09-30 防字幕修复回归:
1) 每个含 <d> 的行都挂上 DIALOGUE_NO_TEXT_EN,且只挂一次(幂等);
2) 注入句不含时间码/双引号/字面 <d>;
3) 二次注入不叠加;
4) _inject_dialogue_no_text 对任意脚本(中文/英文/混合)都生效 —— 换脚本可用。
"""
import importlib.machinery, importlib.util, os, re, sys

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "marquee_director")
sys.path.insert(0, PKG)
# 包 __init__ 需要 comfy 环境,这里直接按文件加载 prompt_pack 模块
ld = importlib.machinery.SourceFileLoader("pp", os.path.join(PKG, "prompt_pack.py"))
sp = importlib.util.spec_from_loader("pp", ld)
pp = importlib.util.module_from_spec(sp)
ld.exec_module(pp)

FAIL = []

SAMPLE = """[Shot 2] At 00:02.900 a handheld medium close-up of <Subject 1>. (S1) <d>张三，我就问问……总惦记别人老婆，法律上得啥责？</d> from 00:02.900 to 00:08.400.
[Shot 3] At 00:09.000 she answers. (S2) <d>汝与曹贼何异？</d> from 00:09.000 to 00:10.000, cold.
[Shot 4] No dialogue here, just the room tone."""


def count_rule(text):
    return text.count(pp.DIALOGUE_NO_TEXT_EN)


out = pp._inject_dialogue_no_text(SAMPLE)
n_d = len(re.findall(r"<d>", SAMPLE))
n_rule = count_rule(out)
if n_rule != n_d:
    FAIL.append(f"期望 {n_d} 条禁令,实得 {n_rule}")

# 幂等
out2 = pp._inject_dialogue_no_text(out)
if count_rule(out2) != n_rule:
    FAIL.append(f"二次注入叠加: {n_rule} -> {count_rule(out2)}")

# 注入句纪律
if "00:" in pp.DIALOGUE_NO_TEXT_EN:
    FAIL.append("禁令含时间码,会污染 _talk_spans")
if '"' in pp.DIALOGUE_NO_TEXT_EN:
    FAIL.append("禁令含双引号,过不了 _QUOTE_RE")
if "<d>" in pp.DIALOGUE_NO_TEXT_EN:
    FAIL.append("禁令含字面 <d>,会干扰逐行块扫描")

# 每句: 禁令挂在**同一条台词行内、`</d>` 之后**，且不越过下一个台词块。
# ★ 2026-09-30 契约调整：插入点从「紧贴 </d>」改成「本块之后那截文本的末尾」——
#   台词自己的时间码窗（`from X to Y, …`）必须留在 `</d>` 紧后面，三条纪律句
#   （约 1500 字符）挂在它后面。v23full 实测：纪律句插在中间会把时间码推离
#   台词块 1300+ 字符，模型于是不遵守时间码（S02 台词提前 1.1s 说完）。
for line in out.splitlines():
    if "<d>" in line:
        after = line.split("</d>", 1)[1]
        if pp.DIALOGUE_NO_TEXT_EN not in after.split("<d>")[0]:
            FAIL.append(f"禁令未挂在台词块之后同一行: {line[:80]}...")
# ★ 时间码窗必须紧邻 </d>（不能被纪律句隔开）
_TAIL_WINDOW = re.compile(
    r"^\s*from\s+\d{1,2}:[0-5]\d\.\d{1,3}\s+to\s+\d{1,2}:[0-5]\d\.\d{1,3}")
for line in out.splitlines():
    if "</d>" not in line:
        continue
    rest = line.split("</d>", 1)[1]
    if re.match(r"\s*from\s+\d", rest) and not _TAIL_WINDOW.match(rest):
        FAIL.append(f"时间码窗未紧邻 </d>: {line[:80]}...")

# 台词块外的内容不被改动(无台词行原样)
if "[Shot 4] No dialogue here, just the room tone." not in out2:
    FAIL.append("无台词行被意外改动")

# 引擎级范本:_inject_lipsync / _inject_dialogue_once 一起挂时不互相吞
out3 = pp._inject_lipsync(SAMPLE)
out3 = pp._inject_dialogue_once(out3)
out3 = pp._inject_dialogue_no_text(out3)
if count_rule(out3) != n_d:
    FAIL.append("三注入器共存时禁令数量不对")
if "DELIVERY: Speak this line EXACTLY ONCE" not in out3:
    FAIL.append("三注入器共存时交付纪律丢失")

# to_shots_json 出口幂等(用最小 fields 走一遍 segment_prompt + 注入)
seg = {"id": "s1", "new_seconds": 8.0,
       "fields": {name: ("X" if name != "detailed_description" else SAMPLE)
                  for name in pp.FIELD_ORDER}}
js = pp.to_shots_json([seg], "t")
shot_text = js["shots_info"][0]["shot"]
if count_rule(shot_text) != n_d:
    FAIL.append(f"to_shots_json 出口禁令数量 {count_rule(shot_text)} != {n_d}")

# ===========================================================================
# 2026-09-30 第四轮(回滚记录):曾把块内 `[语言]` 标签降级成块外散文
#   `spoken in Chinese: <d>…</d>`,理由是"标签触发上屏"。
#   **已回滚**,理由三条(见 prompt_pack.py 里同名注释):
#     ① 实验证伪 —— v25sub 块内已无标签,seg_02 照样烧字;
#     ② 官方 §4.4 原文要求语言标签**在块内**,挪出去才是偏离规范;
#     ③ 会打断 QC 台词正则 `\(S(\d)\)\s*<d>` —— 实测闸门退化成"未解析到台词"。
#   这里保留一条**反向**断言,防止有人再改回去:
# ===========================================================================
TAGGED = ("[Shot 3] At 00:03.200 she speaks. (S2) <d>[Chinese] 汝与曹贼何异？</d> "
          "from 00:03.200 to 00:05.200, cold.")
_seg2 = {"id": "s2", "new_seconds": 8.0,
         "fields": {name: ("X" if name != "detailed_description" else TAGGED)
                    for name in pp.FIELD_ORDER}}
_shot2 = pp.to_shots_json([_seg2], "t")["shots_info"][0]["shot"]
# ① 交付产物必须保留块内语言标签(官方 §4.4)
if not re.search(r"<d>\s*\[Chinese\]", _shot2):
    FAIL.append("交付产物丢了块内语言标签(官方 §4.4 要求块内保留)")
# ② (Sx) 与 <d> 必须紧邻 —— QC 台词正则 \(S(\d)\)\s*<d> 依赖它
if not re.search(r"\(S2\)\s*<d>", _shot2):
    FAIL.append("(Sx) 与 <d> 不再紧邻 —— 会打断 QC 台词解析")
# ③ 防字幕禁令仍要紧贴 </d>
_m = re.search(r"<d>.*?</d>", _shot2, re.S)
if not _m or not re.match(r"^\s*AUDIO-ONLY", _shot2[_m.end():_m.end() + 16]):
    FAIL.append("防字幕禁令没有紧贴 </d>")
# ④ 「缺一不可」校验:块内标签形态算已声明,真缺才报
if pp._dialog_missing_lang("(S2) <d>[Chinese] 甲</d>") != 0:
    FAIL.append("_dialog_missing_lang 对块内标签形态误报")
if pp._dialog_missing_lang("(S2) <d>没写语言</d>") == 0:
    FAIL.append("_dialog_missing_lang 对真正缺语言的台词漏报")

print("FAIL:", FAIL if FAIL else "无 —— 全部通过")
print("--- 注入后台词行示例 ---")
for line in out.splitlines():
    if "<d>" in line:
        print(line[:200], "...")
        break
sys.exit(1 if FAIL else 0)
