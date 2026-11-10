# 自检报告 · 2026-10-18

对象：`custom_nodes/ComfyUI_Marquee_Director` 工作区**未提交改动**
（LUFS 分段响度、双时钟段长网格 39+51n、Director 画布控件 width/height、前端输出规格面板）
范围：全量回归 → 逐文件 diff 审阅 → 数学/契约交叉验证 → 修复复验

---

## 一、结论

| 层 | 方法 | 结果 |
| --- | --- | --- |
| 回归 | `tests/` 19 个 + `tools/test_*.py` 9 个 | 修复后 **28/28 全绿**（初跑 4 红） |
| 专项 | `check_audio_master.py`（7 组检查） | 修复后 **exit 0**（初跑加载即崩） |
| 专项 | `check_workflow.py` | 修复后 **all clean**（初跑 2 问题） |
| 静态 | ruff F/E9/B/SIM/RET/PIE | **无 F821、无 E9**；F 类只剩 10-17 已判定为非缺陷的存量基线 |
| 数学 | `chain_length` 网格全扫（-100…4000） | 输出 100% 落在 17k+5 ∩ 3 倍数 |
| 同源 | `prompt_pack.chain_seconds` vs `common.chain_length` | 0.001s…16s 共 **16000 个采样点逐帧一致，零偏差** |

共修 **1 个 P1、5 个 P2**，清理 3 处 P3。

---

## 二、P1 · `_join_audio` 全部段读最后一个段的 `.lufs`（已修）

**位置** `marquee_director/video_io.py` `_join_audio`（今日改动引入）

```python
for path, skip in parts:            # 第一段循环，path 是循环变量
    ...
    pieces.append((cut, want_seconds))   # ← path 没有随 pieces 传递

for piece, seconds in pieces:       # 第二段循环
    target = _read_lufs_target(path)     # ← path 是第一段循环的遗留变量
```

第二段循环里的 `path` 恒为 **parts 最后一段**的路径。后果：所有段共用末段的
响度目标 —— 末段是对白（-16 LUFS）时，前面的战场段（-8 LUFS）会被
`_is_mastered_tensor` 判成「没母带过」，二次压到 -16，**恰好抹平这次分段响度
改动要引入的动态**；反方向则会跳过真正需要补母带的段。

**修法**：`pieces` 改三元组 `(path, cut, seconds)`，第二段循环逐段解包，
`.lufs` 各读各的。`channels` 生成器同步改 `for _, p, _ in pieces`。

---

## 三、P2 · `save_clip` 残留上一 take 的 `.lufs`（已修）

**位置** `marquee_director/video_io.py` `save_clip`

原实现只在 `lufs_target is not None` 时写伴生文件。同一 `seg_NN.mp4` 重渲而
本次**没有**母带化（无音频 / `MARQUEE_AUDIO_MASTER=0` / 异常跳过）时，上一
take 的 `.lufs` 残留，join 会把旧目标钉在新音频上 —— 与帧边车注释里已有的
同一条原则（「不删旧边车 = 上一 take 的像素服务这一 take 的 mp4」）。

**修法**：新增 `_drop_lufs_target`，`lufs_target is None` 时删除残留伴生。

---

## 四、P2 · `check_audio_master.py` 两处独立加载器被相对导入打死（已修）

**位置** `tools/check_audio_master.py` `load_audio_master` / `_load_with_env`

`audio_master.py:124` 的 `from .common import log, warn`（HEAD 已有）让
`spec_from_file_location` 的裸加载抛
`ImportError: attempted relative import with no known parent package`。
真实 `common.py` 顶层 `import torch` + `comfy.ldm`，整包导入违背工具
「离线可用」的立身之本。

**修法**：合成父包 `audio_master_check_pkg` + 桩 `common`（print 级
`log`/`warn`，保持 printf 风格签名），两个加载器共用（`_load_with_env`
每个环境变量值重载一次，桩注册做了幂等）。

---

## 五、P2 · 测试 5「segment spread」断言旧契约（已修）

**位置** `tools/check_audio_master.py` `check_join_path`

断言「每段收敛到 `TARGET_LUFS`(-13)」与今日的**有意设计**冲突：
`master_segment` 现在按 `profile_target` 分段落 -8/-13/-16。四个探针段
频谱相同 → 全部归类对白 → 落 -16.00 → spread=0.00 但离 -13 差 3 dB，
判 FAIL。是测试过时，不是产品回归。

**修法**：断言改为「每段落在**它自己被分类的目标** ±0.4」，spread 收敛
意图保留（同谱段仍须互相收敛）。复验 exit 0。

---

## 六、P2 · widget 契约清单与工作流 JSON 未同步（已修）

* `tools/test_refine.py`：`width/height` 确实**追加在 schema 末尾**（符合
  append-only，实测顺序 `…seg_cache_codec, width, height`），但期望清单没有
  按测试自身要求复制。补上两个名字 → PASS。
* `workflows/h3_director_unlimited_storyboard.json`：node 700 (H3Director)
  array 23 值对 25 槽、named copy 缺 `width/height`。用自带
  `check_workflow.py --fix` 按 schema 槽位补默认值 0 → 复验 all clean。
  默认 0 = 跟随上游，既有工作流行为逐比特不变。

---

## 七、P3 清理（已做）

| 位置 | 内容 |
| --- | --- |
| `nodes.py` | 删除死代码 `_settings_for_script`（director 改用 `_resolve_canvas` 后全包零调用者，grep 取证） |
| `audio_master.py` | 删除未使用的 `warn` 导入（F401，HEAD 已存在） |
| `check_audio_master.py` | 新引入的 `zip()` 补 `strict=True`（B905） |

---

## 八、验证为非缺陷的项（今日改动内）

* `_align_canvas(0/None) → 0`：默认「跟随上游」路径不会被对齐成 32；
  `stride=None` 延迟解析避开 `CANVAS_STRIDE` 定义顺序的 NameError（注释已说明）。
* `master(normalize=False)`：最终 ceiling 钳制在 `if normalize:` 块**之外**，
  join 只关归一化、不关安全钳，实测成片 pass -13.00 LUFS / -4.80 dBTP。
* 静音段（无音频）走 join：`profile_target` 对全零返回对白档，`master` 对
  不可测信号原样返回 —— 不会崩、不会放大底噪。
* `segment_key` 含 `width/height`（session.py:318-319 实读确认），画布改动
  必然重渲，与 tooltip 承诺一致；`clear_vram` 不在键内，与 10-17 契约一致。
* `_shot_plan`/`_segment_from_widgets` 的 `on_both_clocks` 警告在
  `chain_length` 恒成立的前提下不可达 —— 保留为契约绊线（有人改回
  `generation_length` 时立即报警），非死代码缺陷。
* 回放/锚窗仍走 `guide_length_up`(17k+5)：39 帧本就在双时钟网格上，
  不应被抬到 90；`chain_seconds` docstring 明确禁止误用。
* routes.py 漂移判据（两边非空才判 drift）、session.py 设备名兜底、
  translate_nodes.py `nonlocal n_skip`：10-17 修复全部在位，回归通过。

---

## 九、未覆盖（诚实声明）

1. **真机渲染**：-8/-16 分段响度的实际听感、`chain_length` 吸附后接缝音画
   对齐的真实收益，需 GPU 出片验证（合成信号上数学已证）。
2. **前端面板**：`web/js/h3_director.js` 新增的 `dirWH`/`aspectOf`（相对容差
   4%）/输出规格写回逻辑，只做了静态审阅，未做浏览器内实测。
3. **真实会话目录**上的 `.lufs` 伴生生命周期（重渲/清理/续渲组合）。

---

## 十、如何复验

```bash
PY="E:/Comfy-Desktop/ComfyUI/ComfyUI/.venv/Scripts/python.exe"   # 必须带 torch 的这个
export PYTHONIOENCODING=utf-8                                     # GBK 控制台会崩 ✓/✗

for f in tests/test_*.py tools/test_*.py; do "$PY" "$f" || echo "FAIL $f"; done
"$PY" tools/check_audio_master.py --no-ffmpeg   # exit 0
"$PY" tools/check_workflow.py                    # all clean
"$PY" -m ruff check --select F,E9 marquee_director/   # 只剩存量基线
```

> 注：`E:\Comfy-Desktop\ComfyUI\standalone-env\python.exe` **没有 torch**，
> 用它跑节点测试会大面积 ModuleNotFoundError —— 环境问题，不是代码问题。

---

# 增补 · 第二轮（实机日志取证）

对象：汴京军议_v17 十段实机渲染日志 + 会话目录。第九节「未覆盖」的第 1、3 项
由此部分覆盖（真机出片了；`.lufs` 生命周期在真实清理下暴露了新缺陷）。

## 十一、当日修复的实机验证（全部通过）

* **分段响度**：10 个 `.lufs` 伴生 = -8 / -13 / -16×3 / -13×5，边车逐段实测
  偏差 ≤0.76 LU；join「9/10 命中各自目标」—— 若是修复前的 P1（全部段读末段
  目标），seg_01(-8)/03/04/05(-16) 会被压平到 -13，日志会是 ~5 段重处理。
* **双时钟段长**：141/345/243/243/294/192×3/345/141 全部 ≡39 (mod 51)，
  总 1977 帧 = 82.38s 与 manifest 一致。
* **成片动态**：4s 窗响度跨度 **13.1 LU**（-20.7…-7.6），综合 -13.09 LUFS
  —— 「轻—重呼吸」的设计目标在成片上成立。

## 十二、成片解码真峰 -0.50 dBTP —— 查实为**文档化取舍**，非缺陷

段级 WAV 真峰全部 ≤ -1.00，AAC（130kbps/32kHz）解码过冲 +0.50 → -0.50。
`CEILING_DBTP` 注释当时已实测并拍板：过冲 +0.6~1.1 dB、成片「约 0 dBFS、
与参考片同位」、「每一 dB ceiling 都是响度」。**不加 headroom，维持原决定。**

## 十三、P2 · `master()` 末端钳制在采样峰域、日志冒名 dBTP（已修）

成片趟 `normalize=False` + `pre_limit=False` 时真峰限幅器**一次都不跑**，旧
末端钳只看 `abs(y).max()`；v17 恰好全段 save 时已限幅才没出事（异常路径
漏网的原始段会零保障穿过）。修：末端钳改真峰域（先 limiter 只花峰值，残余
才整轨缩），日志复用同一真峰测量。钉死：fs/4 余弦探针（采样峰 0.6223 <
ceil 0.8913 < 真峰 0.9606）过四关全关的 master → 0.8913，强制生效。

## 十四、P2 · join 段级 rematch 对已母带段**双重**高通+整形（已修）

v17 实际发生：seg_10 裁剪后漂到 -14.33（超 1.0 容差）→ 被 `hp 40 Hz +
shape on` 重处理 —— 它 save 时已经过高通与整形。成片趟注释明文警告过双重
的后果（两级 2 阶 40Hz 串联真峰 +1.1dB、2.5dB 搁架→5dB、3.5dB 凹陷→7dB），
段级却没人挡。修：带 `.lufs` 伴生的段 rematch 只做**电平**校正，无伴生的
原始段才走完整处理。

## 十五、P2 · 清理扫掉伴生后，重建回退分档**误判**（已修）

成片 15 分钟后清理按用户契约只留 seg mp4+成片+storyboard.json。此后重建
音轨走「无伴生 → profile_target 分类」回退，实测分档大错：seg_10（-13 中性）
被判战场档回来**烫 +5.9dB**（-8.41），seg_03/04/05 被拉高 +3.4dB，成片漂到
-12.62 LUFS —— 旧注释「绝不破坏成片」被证伪（高通+整形已改过频谱，分类器
看到的不是它判定的那种原始信号）。修三处：

* `audio_master.snap_target`：无伴生先**吸附**最近规范档（±2.0 LU，吸收裁剪
  漂移+一代 AAC，实测最大 1.4；相邻档距 ≥3.0），命中视同已母带；吸不上才
  是真原始段、正常分类。契约测试 5 项全过。
* `cleanup._is_keep` 保留 `.lufs`（7 字节伴生元数据，PANEL_KEEP 同理）。
* `rebuild_film_audio.py` 报告列改走与 join 完全同一条决策链。

复验（v17 真实目录）：10 段全部 `via=snap` 吸回正确档位（与被扫掉的 `.lufs`
逐段一致），9 段放行、seg_10 仅电平 +1.33dB；成片 **-13.08 LUFS / 真峰
-1.00 dBTP**。回归 28/28、check_audio_master exit 0、ruff 仅存量基线。

## 十六、v17 返工产物与取舍

`汴京军议_EP001/汴京军议_v17_remastered.mp4`（视频流逐比特 copy，放在剧本夹
——清理只扫会话目录，此处安全）。A/B 实测：

| | 原成片 | 重混版 |
| --- | --- | --- |
| 综合响度 | -13.09 LUFS | -13.14 LUFS |
| 动态跨度（4s 窗） | 13.1 LU | 13.0 LU |
| seg_10 低频占比 | 25.96% | **22.43%**（双重搁架已除） |
| seg_9 低频占比（对照） | 16.52% | 16.68%（其余段未动） |
| 解码真峰 | -0.50 dBTP | -0.69 dBTP |

代价如实说：边车已被清理扫掉，重建音频比原成片**多一代 AAC 解码**（原成片
join 读的是无损 WAV 边车）。seg_10 音色正确 vs 其余九段多一代有损 —— 两部
成片都保留，选用哪部由用户定。`_film_audio.wav` 是中间产物，留在会话目录、
下次清理会按契约扫掉。

## 十七、日志中两处与本包无关的异常

* `Error: extra_pnginfo[0] is not a dict or missing 'workflow' key` ×2：
  `comfyui-custom-scripts` 的 ShowText 节点（拿不到前端 workflow 元数据时
  打印并跳过嵌入），无害，非本包代码。
* **16 块未译出（已置空）**：翻译 API 失败（英译 0、二次救回 0、逐行救回 0，
  121 块本就是英文）。节点安全行为正确（宁置空不漏中文），但 v17 提示词里
  这 16 个台词块是空的 —— 需检查 Base URL/模型名/Key，重译后重渲受影响段。

---

# 增补 · 第三轮（GitHub 上游与社区方案调研融合）

方法：git 远端比对（fetch 网络超时，改用 GitHub API 拉提交史）+ 逐候选与
本地代码交叉验证，只融合**本地确实缺**的东西。

## 十八、调研结论总表

| 来源 | 候选方案 | 本地核对 | 处置 |
| --- | --- | --- | --- |
| upstream `loopforge0/ComfyUI-H3-Continuous` | PR#3：pre-0.34.0 守卫 + 分段 LoRA | `assert_clip_guides_supported` 已在 common.py；per-shot LoRA 走图接线（nodes.py:369-380）且缓存键含模型指纹 | 已具备，不融合 |
| `AIMixer/ComfyUI_MiniMaxH3_Director` b168de8 | 解码音频真实采样率修复 | 本地 `read_audio` 读 `stream.sample_rate`，无 44100 硬编码；v17 全程 32kHz 正确 | 无此 bug |
| 同上 a8f57b8 | latent 权重 "3d" 子串误筛 | 本地无该模式 | 不适用 |
| 同上 860df2f/d59c5d7 | 段缓存目录 = 文件夹名+节点 id | 本地缓存按**会话目录**隔离（seg_NN.* 边车），不依赖节点 id，多工作流天然不撞 | 不适用 |
| 同上 8151b29 | 段间清理释放预取队列 | 本地已借鉴（commit 1fc51f7） | 已在位 |
| 同上 4c6245e | 导演包导入/导出进度 | 本地走 `send_sync` 事件流（`emit_progress`），机制不同且已覆盖 | 不采 |
| `fblissjr/ComfyUI-h3-explorations` | h3_rules 最新（10-15：362 上限、链长算出不写死） | 与本地 `chain_length` 同余式实现一致；近期 decisions.md 全在 masked-v2v 域 | 一致，无新规则 |
| ComfyUI 核心 02aa707 | prefetch race 修复 | 只改核心文件，本地 core 0.42.1 已含；本包 te_prefetch shim 包装的是文本编码器预取（另一机制） | 已在位 |
| **ComfyUI 核心 b16023b** | **移除 torchaudio 依赖** | **本地命中：`_join_audio` 重采样路径 `import torchaudio`** | **已融合修复，见十九** |
| `luxu1999/…ChainDirector` | 首帧锁定 + 自动拼接 | 技术路线不同（本地 latent 回放 + 漂移校正更强），8 月底后停更 | 无借鉴点 |
| `alianess/…director-prompts` | 帧验证 | README 两次拉取超时；该方向本地已有 face_qc / subtitle_qc / 接缝差异 | 不追 |

## 十九、P2 · join 重采样依赖已被核心移除的 torchaudio（已修）

**位置** `video_io.py _join_audio`（全包唯一 `import torchaudio` 调用点）

**取证链**：核心 b16023b（2026-09-22）移除 torchaudio 依赖；本地 core
0.42.1 的 `nodes_minimax_h3.py`/`nodes_audio.py` 已零引用；当前 venv 里
torchaudio 2.11.0 还在，但 requirements 不再钉它 —— Desktop 下一次重建
环境它就消失。届时混合采样率段（如无损边车 32kHz + 某段 mp4 44.1kHz）会让
整条成片路径崩在 `import torchaudio`，报错点离根因很远。v17 全段 32kHz
同率，**没触发只是没踩到**。

**修法**：新增 `_resample_waveform` —— `scipy.signal.resample_poly`
（gcd 化简 44100→32000 = 320/441 多相 Kaiser FIR，与 torchaudio 实现同类；
scipy 已是 `audio_master` 硬依赖，不新增依赖面），dtype/device/batch 维保留。

**钉死**：`tests/test_resample.py`（常驻，7 项）—— 源码不得 import
torchaudio（只禁 import 语句，docstring 提及不违约）、长度按比率（±2 样本）、
440Hz 主频 ±1Hz、幅度 ±5%、dtype/batch 保留、同率等长副本。

**复验**：回归 **29/29** 全绿（含新测试）、check_audio_master exit 0、
check_workflow all clean、ruff F/E9 仅存量基线。

## 二十、第三轮未覆盖（诚实声明）

1. `_resample_waveform` 只做了合成正弦验证；真实混合采样率会话的端到端
   join 无现成样本（v17 全程同率），逻辑上与旧 torchaudio 路径等价替换。
2. 调研以提交史+README 为主，未逐行读 AIMixer/luxu1999 的实现源码；
   判断「本地已具备/不适用」的依据是本地代码交叉验证，不是对方代码审阅。
