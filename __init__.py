"""ComfyUI_Marquee_Director: seamless multi-segment MiniMax H3 video.

Chain any number of H3 renders into one unbroken take. Each segment opens on an
exact replay of the previous segment's last frames, pinned with MiniMaxH3AddGuide
rather than merely conditioned on -- which is the difference between a joined shot
and a cut.
"""

from .marquee_director import MarqueeDirectorExtension

# 文本编码器的权重预取：上游在编码 prompt 时（还没有 KV cache）把预取关掉了，
# 25.88 GB 的 TE 只能逐层串行流式读，实测每段浪费 9–10 分钟、占整条链 35 分钟。
#
# ★ 用「运行时 shim」而不是改 comfy/** 的源码 —— 核心文件改不动：
#   2026-09-23 07:49 那次重启，Desktop 自动更新（v0.37.0 → v0.37.1）把
#   llama.py 还原成上游原版，连旁边的 .bak 都删了。
#   shim 装在本包内，更新不会碰它；上游自己修好后它会自动不再生效。
#   关掉：MARQUEE_TE_PREFETCH=0
try:
    from .marquee_director import te_prefetch
    te_prefetch.install()
except Exception:      # 绝不让一个性能补丁挡住节点加载
    pass

# Director 的前端面板（PACK 模块 / 参考图分类 / 分镜时间线 / 小马进度）。
# ComfyUI 在加载自定义节点时读这个属性，把目录挂到 /extensions/ 下。
WEB_DIRECTORY = "./web"

async def comfy_entrypoint() -> MarqueeDirectorExtension:
    # 路由在 nodes.py 里注册过一次；这里是第二次机会 —— 万一那次因为
    # PromptServer 还没建好而失败，这里再试一次，面板照样有实时数据。
    try:
        from .marquee_director import routes
        routes.register()
    except Exception:
        pass
    # 预取 shim 是幂等的，这里再试一次，兜住上面那次 import 失败的极端情况。
    try:
        from .marquee_director import te_prefetch
        te_prefetch.install()
    except Exception:
        pass
    return MarqueeDirectorExtension()
