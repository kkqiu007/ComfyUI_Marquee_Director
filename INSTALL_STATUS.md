# 安装与模型清单

## 节点包

- 位置：`ComfyUI/custom_nodes/ComfyUI_Marquee_Director/`
  —— 目录名**不要改**，ComfyUI 靠目录名识别包。
- 无第三方依赖，不需要 `pip install`。
- 需要 **ComfyUI 0.34.0 或更新**（更早的版本根本没有 MiniMax H3 节点）。
- 装好后的自检：重启 ComfyUI，双击画布输入 `H3 Chain`，能搜到 **H3 Chain Settings** 即成功。

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

`workflows/h3_continuous.json` 里每个 loader 节点的 `properties.models` 都带着下载
地址（ComfyUI 官方模板的做法）。打开工作流时若有文件缺失，ComfyUI 会直接提示
要不要替你下载，比手动放文件省事。

---

## 输出目录

渲染产物落在：

```
ComfyUI/output/h3_continuous/<会话名>/
```

| 文件 | 说明 |
| --- | --- |
| `seg_01.mp4`, `seg_02.mp4`, … | 各段独立输出 |
| `seg_XX.tail.*` | 各段结尾帧，下一段靠它对齐接缝 |
| `manifest.json` | 本次渲染的记录，含每段的指纹 |
| 成片 | 由 `H3 Chain To Video` 拼出的完整一条 |

`<会话名>` 来自 **H3 Chain Settings** 的 `session_name`。
改 `session_name` 就是开一个全新会话，旧的段文件不会被复用。
