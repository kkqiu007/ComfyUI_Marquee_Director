"""ComfyUI_Marquee_Director: seamless multi-segment MiniMax H3 video.

Chain any number of H3 renders into one unbroken take. Each segment opens on an
exact replay of the previous segment's last frames, pinned with MiniMaxH3AddGuide
rather than merely conditioned on -- which is the difference between a joined shot
and a cut.
"""

from .marquee_director import MarqueeDirectorExtension

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
    return MarqueeDirectorExtension()
