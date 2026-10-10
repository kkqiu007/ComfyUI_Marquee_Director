# 安装与模型清单

## 节点包

- 位置：`ComfyUI/custom_nodes/ComfyUI_Marquee_Director/`
  —— 目录名**不要改**，ComfyUI 靠目录名识别包。
- 无第三方依赖，不需要 `pip install`。
- 需要 **ComfyUI 0.34.0 或更新**（更早的版本根本没有 MiniMax H3 节点）。
- 装好后的自检：重启 ComfyUI，双击画布输入 `H3Chain`，能搜到
  **H3 链路设置** 即成功。
  本包现行注册的节点共八个（ID ↔ 中文名 的完整对照见 `README.md` 文首）：
  `H3ChainSettings` / `H3PromptPackParser` / `H3Director` /
  `H3Refine` / `H3FaceRefine` / `H3ScriptTranslate` / `H3StoryboardSave` /
  `H3StoryboardLoad`。其中 `H3Refine` / `H3FaceRefine` / `H3ScriptTranslate` 是
  **可选外接**，不接线就不参与渲染。

> **2026-10-10 变更**：节点 ID 全量统一成 `H3*`，**不保留旧别名**。
> `MinimaxH3SaveJson` / `MinimaxH3LoadJson` 已改名为 `H3StoryboardSave` /
> `H3StoryboardLoad` —— 这不是为了整齐：另一个已安装的包
> `Comfyui-MinimaxH3-Ref2VA-Auto-Video` **同样注册了这两个旧 ID**，谁先加载谁赢，
> 于是旧图里的"剧本存档"节点会静默变成别人家那个（槽位完全不同）。
> 旧工作流用 `python tools/check_workflow.py --fix` 改写 `type` 即可。

> **2026-09-20 变更**：`H3ScriptBatchRender` / `H3ScriptRepairSegment` /
> `H3ChainToVideo` / `H3LoadSession` / `H3SegmentTimeline` / `H3ShotPrompt` /
> `H3ShotsBoard` / `H3ShotRenderer` 已下线（全部被 `H3 导演` 取代）。
> 原先随包分发的 `workflows/h3_continuous.json`、
> `workflows/h3_continuous_local_prompt.json`、`demo_8shots_workflow.json`
> 引用了这些节点，已一并移除。现行样例工作流是
> **`workflows/h3_director_unlimited_storyboard.json`**。

---

## 模型清单（合计约 42.5 GB）

| 用途 | 文件 | 放置目录 | 大小 |
| --- | --- | --- | --- |
| Diffusion | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` | ~21 GB |
| Text encoder | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` | ~15 GB |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` | ~2 GB |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` | ~2 GB |
| Turbo LoRA（强烈建议） | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | `models/loras/` | ~0.5 GB |

> 必须是 **ref2va** 模型，不是 `fl2va`。本包串联的是**参考图条件**渲染，
> `fl2va` 不吃参考图。
>
> Turbo LoRA 可选但几乎必装：把采样从 20 步压到 4 步，
> 在 12 GB 卡上约 9 分钟一段，而不是 31 分钟。

### 下载

在 **ComfyUI 根目录**下执行（路径是相对的，别照抄成绝对路径）：

```bash
BASE=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main

curl -L "$BASE/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
     -o models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors

curl -L "$BASE/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
     -o models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors

curl -L "$BASE/vae/minimax_h3_video_vae_fp16.safetensors" \
     -o models/vae/minimax_h3_video_vae_fp16.safetensors
curl -L "$BASE/vae/minimax_h3_audio_vae_fp32.safetensors" \
     -o models/vae/minimax_h3_audio_vae_fp32.safetensors

curl -L "$BASE/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
     -o models/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
```

**国内镜像**：把上面 `BASE` 里的 `huggingface.co` 换成 `hf-mirror.com` 即可，
其余路径完全一致。

Diffusion 模型 21 GB，建议用支持断点续传的工具（`curl -C -`、IDM、aria2）——
裸 `curl` 断一次就得重来。

### 也可以交给 ComfyUI 自己拉

`workflows/h3_director_unlimited_storyboard.json` 里 H3 链路上的五个 loader 节点
（UNET / 文本编码器 / 视频 VAE / 音频 VAE / Turbo LoRA）的 `properties.models`
都带着下载地址（ComfyUI 官方模板的做法）。打开工作流时若有文件缺失，ComfyUI 会直接提示
要不要替你下载，比手动放文件省事。

第六个 loader（`CLIPLoader`，喂 **H3 剧本英译**）没有带 —— 它只在英译节点
切到 `local` 时才被读，工作流默认走 `openai_api`，用不到。

---

## 输出目录

根目录**跟随 ComfyUI 自己的配置**（`folder_paths.get_output_directory()`，全包只有
`session.output_root()` 这一处解析它）。Desktop 改 `outputDir` / 命令行加
`--output-directory` 都会带着产出一起走 —— 以前包里有四处各自拼根，改一次设置就把
产出分到几棵树里，表现就是「面板说会话不存在」和「断点渲染又从第 1 段开始」。

```
<ComfyUI 输出目录>/h3_continuous/<剧本名>/<版本名>/     现行
<ComfyUI 输出目录>/h3_continuous/<会话名>/              升级前的单层布局，原位沿用
```

| 文件 | 说明 |
| --- | --- |
| `seg_01.mp4`, `seg_02.mp4`, … | 各段独立输出 |
| `seg_XX.tail.*` | 各段结尾帧，下一段靠它对齐接缝 |
| `seg_XX.frames.mkv` | 无损像素边车，只在「段缓存像素 = ffv1」时写 |
| `manifest.json` | 本次渲染的记录，含每段的指纹（**只记文件名，不记绝对路径**） |
| `storyboard.json` | 「H3 剧本存档」写的清洗后分镜 |
| `<版本名>.mp4` | 成片，名字永远等于它所在的目录名 |
| `../h3_seg_export/<时间戳>/` | 「逐段导出」的交付物，**故意在 `h3_continuous` 之外**（清理不碰它） |

`<剧本名>` 取 PACK 头部的 `Project`（源视频模式取视频文件名；也可在 **H3 链路设置**
的「剧本名称」里手填），`<版本名>` 取 `session_name`。两个都空才落到
`h3_continuous/h3_pack/` —— 全包唯一的兜底名。

★ **版本名留空 + 关掉「复用」= 自动 `r1`/`r2`…，同一部剧本的每次全新渲染各占一个
文件夹**，上一版的产出原封不动。开「复用」时正相反：一定接在最新那一版上。

★ **老的单层目录不搬**：里面已有 18 段真实产出（本机 `茜色` 就是这么一条），搬进
两层布局等于让断点续渲看不见它们 —— 凭空一次全量重渲。`resolve_session_dir()`
先认旧目录，只有它没有产出（或本次明确重开一轮）才落到两层布局。
