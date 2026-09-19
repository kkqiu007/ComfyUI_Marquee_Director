/*
 * ComfyUI-Marquee-Director 面板：PACK 解析 / 参考图 / 分镜时间线（含奔马动画）。
 * 四个模块都挂在 H3Director 节点上，共享同一份 PACK / 参考图 / 进度数据。
 * 数据来源：PACK 解析 POST /h3/pack_preview；进度来自 ws 事件
 *          h3.director.progress（Director 每段开始/结束各广播一次）。
 * 版本水印见文件末尾 console.log —— 看不到就 Ctrl+Shift+R 强刷清缓存。
 */

/* ------------------------------------------------------------------ */
/* app / api 解析                                                      */
/* ------------------------------------------------------------------ */
/* 不从 scripts/app.js 静态 import：1.51.x 的 shim 在模块求值那一刻读
 * window.comfyAPI，未挂上时即抛 TypeError、整模块静默失效。
 * 改为从 window 上取，取不到用 null 顶着，后台异步补 shim 只作兜底。 */
function _fromWindow(kind) {
  try {
    if (typeof window === "undefined") return null;
    const capi = window.comfyAPI && window.comfyAPI[kind];
    if (capi && capi[kind]) return capi[kind];
    if (window[kind]) return window[kind];
  } catch (e) { /* 拿不到就返回 null */ }
  return null;
}

let app = _fromWindow("app");
let api = _fromWindow("api");

const EXT = "ComfyUI.H3Continuous.Director";

console.log("[H3 Director] 扩展已载入 — app:%s api:%s",
            !!(app && app.registerExtension), !!api);

// 后台补一份 shim 的值（仅当 window 上没拿到时才有意义）；用动态 import，
// 失败只是打条 warn，不会像静态 import 那样把整个模块炸在第一行。
(async () => {
  try {
    if (!app) app = (await import("../../scripts/app.js")).app || app;
    if (!api) api = (await import("../../scripts/api.js")).api || api;
  } catch (e) {
    console.warn("[H3 Director] scripts/*.js shim 不可用，只用 window 上的 app/api：", e);
  }
})();

/** 面板构建失败时，把错误直接画在节点上（不依赖开 devtools 才能看到）。 */
function showPanelError(node, err) {
  console.error("[H3 Director] 面板构建失败：", err);
  try {
    const box = document.createElement("div");
    box.style.cssText = "padding:6px 8px;border:1px solid #a33;border-radius:6px;"
      + "background:rgba(120,20,20,.35);color:#fdd;font:11px/1.5 system-ui;"
      + "white-space:pre-wrap;word-break:break-all;";
    box.textContent = "ComfyUI-Marquee-Director 面板加载失败：\n" + (err && err.message ? err.message : String(err))
      + "\n（详细堆栈见浏览器控制台）";
    node.addDOMWidget("h3_director_error", "div", box, {
      serialize: false,
      hideOnZoom: false,
      getValue() { return ""; },
      setValue() {},
    });
  } catch (e) { /* 连错误框都画不出来就只能靠控制台了 */ }
}

/* ------------------------------------------------------------------ */
/* 样式：面板不画独立外框，透明底、无边框，只靠细横线分区，宽吃满节点。 */
/* ------------------------------------------------------------------ */
const CSS = `
/* ========= 双模式对齐：Nodes 2.0 多一层 .lg-node-widget 栅格（窄 4px），
   这里补回，使两种模式下面板宽度一致。 ========= */
.lg-node-widget .h3d { margin-right: -4px; }

/* ============================================================
 * ★ 第 18 轮：Liquid Glass 设计令牌层
 * ------------------------------------------------------------
 * 第 17 轮把参考图卡片做成了 Liquid Glass，但参数全散在那一块的 CSS 里。
 * 现在要把同一套语言铺到**所有版块**，第一件事就是把这些数字提成令牌 ——
 * 否则 20 个版块各写各的 rgba，颜色/模糊早晚漂移，改一处要改二十处。
 *
 * 令牌分四组：
 *   --h3d-glass-*  玻璃材质（模糊/亮度/饱和/底色）
 *   --h3d-rim-*    边缘高光（specular rim）—— Liquid Glass 的招牌厚度感
 *   --h3d-tint-*   语义色（四类分类 + 危险/主色），统一用"玻璃 tint"手法
 *   --h3d-r-*      圆角刻度（从卡片到胶囊，一套递增关系）
 *   --h3d-ease     Apple 偏爱的动效曲线
 *
 * ★ 为什么令牌里带 alpha 的色值要写整条 rgba() 而不是只存通道：
 *   CSS 变量在 rgba(var(--x), .5) 这种写法里做算术不可靠（旧浏览器/部分内核），
 *   直接存整条 rgba() 最稳。代价是同一色相的多个透明度要各存一条。
 * ============================================================ */
.h3d {
  /* ---- 玻璃材质（regular 变体）----
   * Apple 原文："The regular variant blurs and adjusts the luminosity of
   * background content to maintain legibility." → blur + brightness 是这个
   * 变体的定义性特征，缺一不可。brightness 尤其关键：只模糊不调亮度，
   * 浅色背景上的白字照样看不清。 */
  --h3d-glass-blur: 12px;
  --h3d-glass-bright: .94;      /* <1 压暗，保证上面白字可读 */
  --h3d-glass-sat: 165%;        /* >100% 吸背景色，让玻璃"染"上底下的颜色 */
  --h3d-glass-bg: rgba(22, 22, 25, .46);
  --h3d-glass-bg-strong: rgba(16, 16, 18, .62);   /* 大面积面板用，字多要更实 */
  /* Apple 文档里唯一的硬性数字：明亮背景下加 35% 深色变暗层 */
  --h3d-glass-dim: rgba(0, 0, 0, .35);

  /* ---- specular rim 边缘高光 ----
   * 上亮下暗 = 光从上方来 = 玻璃厚度感。数值分两档：
   * 小元件用 .14/.28，大面积浮层用 .20/.30（面积越大高光要越明显才看得出）。 */
  --h3d-rim: inset 0 1px 0 rgba(255,255,255,.14), inset 0 -1px 0 rgba(0,0,0,.28);
  --h3d-rim-lg: inset 0 1px 0 rgba(255,255,255,.20), inset 0 -1px 0 rgba(0,0,0,.30);
  --h3d-rim-hi: inset 0 1px 0 rgba(255,255,255,.28);   /* 按钮等交互件 */
  --h3d-lift: 0 4px 14px rgba(0,0,0,.42);              /* 浮层投影 */
  --h3d-lift-sm: 0 1px 3px rgba(0,0,0,.30);

  /* ---- 交互件（按钮/胶囊）"压印"在玻璃上的填充 ----
   * ★ 关键：这些**不带 backdrop-filter**（避免 glass-on-glass，Apple 明确要求节制）。
   *   它们靠填充+高光在已有玻璃上"压印"出层次。 */
  --h3d-press-bg: rgba(255,255,255,.14);
  --h3d-press-bd: rgba(255,255,255,.26);
  --h3d-press-bg-hi: rgba(255,255,255,.28);
  --h3d-press-bd-hi: rgba(255,255,255,.50);

  /* ---- 语义色（玻璃 tint 手法：只换色相，不换材质）---- */
  --h3d-char:  #6affa0;   --h3d-char-a:  rgba(79,255,143,.42);
  --h3d-prop:  #e2a2f7;   --h3d-prop-a:  rgba(214,140,240,.42);
  --h3d-scene: #a3c9f5;   --h3d-scene-a: rgba(135,182,235,.42);
  --h3d-other: #dcdcdc;   --h3d-other-a: rgba(200,200,200,.34);
  --h3d-warn:  #f5c542;   --h3d-bad:     #ff8b8b;
  --h3d-acc:   #4fff8f;                          /* accent 绿（本包品牌色）*/
  --h3d-info-a: rgba(55,138,221,.55);            /* ⇄ 等"信息类"操作 */
  --h3d-danger-a: rgba(224,75,58,.55);

  /* ---- 圆角刻度 ----
   * Liquid Glass 是 iOS 26 / macOS Tahoe 语言，圆角明显比传统桌面 UI 松弛。
   * 一套递增关系：小控件 6 → 输入 7 → 卡片 10 → 面板 12 → 胶囊 999。 */
  --h3d-r-xs: 6px;
  --h3d-r-sm: 7px;
  --h3d-r-md: 10px;
  --h3d-r-lg: 12px;
  --h3d-r-pill: 999px;

  /* ---- 动效曲线 ----
   * Apple 偏爱的平滑减速曲线，比 ease / linear 更有"物理感"。 */
  --h3d-ease: cubic-bezier(.32, .72, 0, 1);
  --h3d-dur: .18s;
  --h3d-dur-fast: .12s;

  /* ---- 文字在玻璃上的投影 ----
   * 玻璃亮度会随背景变，文字需要一道极淡的暗影"托"起来才稳。 */
  --h3d-txt-shadow: 0 1px 2px rgba(0,0,0,.5);
}
/* ---- 玻璃材质的统一工具类：任何元素加上它就获得 regular 变体 ----
 * 集中一处，改令牌即全站生效。带 -webkit- 前缀是因为 ComfyUI 桌面版
 * 在部分机器上是 WebKit/旧 Chromium 内核。 */
.h3d .h3d-glass,
.h3d-glass {
  background: var(--h3d-glass-bg);
  backdrop-filter: blur(var(--h3d-glass-blur))
                  brightness(var(--h3d-glass-bright))
                  saturate(var(--h3d-glass-sat));
  -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                           brightness(var(--h3d-glass-bright))
                           saturate(var(--h3d-glass-sat));
  box-shadow: var(--h3d-rim), var(--h3d-lift-sm);
}
.h3d .h3d-glass--strong,
.h3d-glass--strong { background: var(--h3d-glass-bg-strong); }

/* ========= 外框：只是排版容器，不画任何背景/边框 ========= */
/* ★ 底部 padding 归零：用户要「背景层紧跟文本框」——节点紫底的下边缘要贴着
 *   textarea 的下边框，中间不留多余空白。此处一旦留白（哪怕是 6px/14px），
 *   视觉上就是"文本框浮在背景上、底下空一截"。
 *   留白的正当来源只剩 textarea 自身的 padding(7px) + border(1px)，那属于
 *   文本框本体，保留。 */
.h3d { font: 11px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif;
       color: #e0e0e0; background: transparent; border: 0;
       padding: 4px 0 0; box-sizing: border-box;
       display: flex; flex-direction: column; gap: 9px;
       min-width: 0; }
.h3d *, .h3d *::before, .h3d *::after { box-sizing: border-box; }

/* ========= 版块：细分隔线分区，不做成独立卡片 ========= */
/* ★ flex: none 必须有：.h3d 是 flex column，默认 flex-shrink:1 会把版块压扁
 *   去凑容器的高度（实测「7 参考图」被从 1116px 压到 148px）。一旦压缩，
 *   root.scrollHeight 量到的就是"被压缩后"的高度而不是内容真实高度 ——
 *   折叠/展开时压缩一解除，内容弹回自然高度，节点高度就一路棘轮上涨
 *   （实测 1381 → 2470 → 2857）。禁掉收缩后 scrollHeight 恒为真实值。 */
.h3d-sec { display: flex; flex-direction: column; gap: 6px; min-width: 0;
           flex: none; }
/* 版块之间：从实线改为**渐隐分隔线** —— 两端淡出比一条硬线更安静，
 * 也和玻璃的柔和边缘语言一致（第 18 轮）。 */
.h3d-sec + .h3d-sec { border-top: 0; padding-top: 9px; position: relative; }
.h3d-sec + .h3d-sec::before {
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 1px;
  background: linear-gradient(90deg, transparent, #2c2c2e 18%, #2c2c2e 82%, transparent);
  pointer-events: none; }
/* 提示词是第一块面板，与上边原生控件保持 8px 间隔（用户要求） */
.h3d-sec--prompt { margin-top: 8px; }

.h3d-hd { display: flex; align-items: center; gap: 8px; min-height: 22px;
          min-width: 0; }
.h3d-hd .h3d-title { font-size: 11px; font-weight: 600; color: #d8d8d8;
                     letter-spacing: .02em; cursor: pointer; user-select: none;
                     transition: color var(--h3d-dur-fast); }
.h3d-hd .h3d-title:hover { color: #fff; }
.h3d-hd .h3d-tag { font-size: 10px; color: #7d7d7d; }
.h3d-hd .h3d-tag.is-ok  { color: var(--h3d-acc); }
.h3d-hd .h3d-tag.is-bad { color: var(--h3d-bad); }
.h3d-hd .h3d-spacer { flex: 1 1 auto; min-width: 0; }

.h3d-bd { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.h3d-bd.collapsed { display: none; }

/* ========= 按钮（第 18 轮：Liquid Glass interactive 玻璃）=========
 * Apple 把按钮归为"控件层"，正是 Liquid Glass 的适用对象。
 * ★ 但按钮**不带 backdrop-filter** —— 它们浮在已经有玻璃的版块上，
 *   再叠一层就是 glass-on-glass（Apple 明确要求节制）。改用
 *   「半透明填充 + 顶部亮线 + 描边」在玻璃面上**压印**出立体感。 */
.h3d-btn { height: 24px; padding: 0 9px; font-size: 10.5px; line-height: 1;
           background: var(--h3d-press-bg); color: #eaeaea;
           border: 1px solid var(--h3d-press-bd); border-radius: var(--h3d-r-xs);
           cursor: pointer; white-space: nowrap;
           box-shadow: var(--h3d-rim-hi);
           transition: background var(--h3d-dur-fast), border-color var(--h3d-dur-fast),
                       color var(--h3d-dur-fast), box-shadow var(--h3d-dur-fast),
                       transform .1s; }
           /* ★ 第 32 轮：ComfyUI 桌面版（Chromium 内核）按钮获得 focus 时默认
              套一圈 2px 蓝色 outline，落点在按钮**外侧**。工具栏 4 个按钮
              紧排时，focus 框会跨按钮边界到右侧按钮上 → 「重渲已改段」
              点不到焦点、用户看到的蓝框在它和「刷新状态」之间。
              改用 inset outline + focus-visible：键盘 Tab 才有 ring（不打扰
              鼠标操作），环贴在 border 内侧，不溢出。 */
.h3d-btn:focus { outline: 0; }
.h3d-btn:focus-visible { outline: 2px solid rgba(79,174,255,.85);
                          outline-offset: -2px; }
.h3d-btn:hover { background: var(--h3d-press-bg-hi);
                 border-color: var(--h3d-press-bd-hi); color: #fff;
                 box-shadow: inset 0 1px 0 rgba(255,255,255,.42), var(--h3d-lift-sm);
                 transform: translateY(-1px); }
/* active：按下去回弹，符合 macOS 物理直觉 */
.h3d-btn:active { transform: translateY(0) scale(.96);
                  box-shadow: inset 0 1px 2px rgba(0,0,0,.32); }
.h3d-btn.is-primary { background: rgba(47,158,68,.78); border-color: rgba(79,255,143,.62);
                      color: #fff; }
.h3d-btn.is-primary:hover { background: rgba(56,176,74,.88);
                            border-color: rgba(79,255,143,.85); }
.h3d-btn:disabled { opacity: .38; cursor: default; transform: none;
                    box-shadow: none; }
/* 批量模式进行中：按钮变绿，提示"再点一次退出" */
.h3d-btn.is-on { background: rgba(47,158,68,.82); border-color: var(--h3d-acc); color: #fff; }
.h3d-btn.is-on:hover { background: rgba(56,176,74,.92); }

/* ========= 1) 项目信息：一行紧凑键值（不再套卡片，直接贴在节点上） ========= */
.h3d-meta { display: flex; flex-wrap: wrap; gap: 3px 18px;
            padding: 1px 0 2px; background: transparent;
            border: 0; border-radius: 0; min-width: 0; }
.h3d-item { display: flex; align-items: baseline; gap: 6px; min-width: 0; }
.h3d-item .k { font-size: 10px; color: #7d7d7d; flex: none; }
.h3d-item .v { font-size: 12px; font-weight: 500; color: #eaeaea;
               font-variant-numeric: tabular-nums;
               overflow-wrap: anywhere; word-break: break-word; min-width: 0; }
.h3d-item .v.is-empty { color: #555; font-weight: 400; }
.h3d-item .v .sub { font-size: 10px; font-weight: 400; color: #7d7d7d; }

/* ========= 2) 参考图（2026-09-16 第 17 轮：Apple Liquid Glass 设计语言）=========
 *
 * 第 16 轮的遮罩式 hover 面板方向已确认正确，本轮按 Apple Liquid Glass 规范
 * 逐条打磨材质本身。规范来源（都是实测查证的，不是凭印象）：
 *   · developer.apple.com/documentation/technologyoverviews/liquid-glass
 *   · developer.apple.com/design/human-interface-guidelines/materials
 *   · WWDC25 Session 219「Meet Liquid Glass」+ 219/356/323 系列
 *
 * ---- 抄过来的四条硬规则 ----
 *
 * ① **玻璃是「漂浮的功能层」，不是内容层材质。**
 *    Apple 原文："Liquid Glass forms a distinct functional layer for controls and
 *    navigation elements ... Don't use Liquid Glass in the content layer."
 *    → 应用在这里：**缩略图本身是内容，绝不上玻璃**（保持纯粹的照片）。
 *      玻璃只给「覆盖在内容之上的功能层」——即 hover 遮罩、操作按钮、批量条。
 *      这是本轮最重要的一条：第 16 轮遮罩用的是"深灰蒙版"，本质是把内容压暗；
 *      Liquid Glass 的做法是**用玻璃浮在内容上**，让缩略图透过玻璃仍然可辨。
 *
 * ② **regular 变体：模糊 + 调整背景亮度（luminosity）以保证可读性。**
 *    Apple 原文："The regular variant blurs and adjusts the luminosity of background
 *    content to maintain legibility."
 *    → 应用：backdrop-filter 用 blur() + brightness() + saturate() 三件套。
 *      brightness 是关键 —— 只模糊不调亮度，浅色照片上的白字仍然看不清。
 *
 * ③ **明亮背景下加 35% 深色变暗层（这是 Apple 文档里唯一的硬性数字）。**
 *    Apple 原文："If the underlying content is bright, consider adding a dark
 *    dimming layer of 35% opacity."
 *    → 应用：遮罩底色用 rgba(0,0,0,.35)，叠在玻璃层之内（不是替代模糊）。
 *
 * ④ **节制使用；不要 glass-on-glass 嵌套。**
 *    Apple 原文："Use Liquid Glass effects sparingly."
 *    实践笔记的检查清单："是否出现 glass-on-glass 的嵌套层叠"
 *    → 应用：**一层玻璃，中间不开洞**。.veil 是唯一玻璃面，
 *      里面的按钮不再各自带玻璃 —— 改用「高光描边 + 半透明白填充」在玻璃上"压印"，
 *      视觉上仍是玻璃的一部分，但没有第二层 backdrop-filter（那会又慢又脏）。
 *
 * ---- 视觉三要素（Liquid Glass 的招牌）----
 *
 *  · **specular rim（边缘高光）**：玻璃边缘有一道 1px 的亮线，模拟真实玻璃
 *    边缘的折射高光。实现在 .h3d-ref::after —— 用 inset 0 0 0 1px 的
 *    **渐变边框**（上亮下暗、左亮右暗），比纯色 border 更像玻璃。
 *
 *  · **lensing（透镜感）**：玻璃边缘会让背后内容轻微位移/放大。
 *    CSS 做不到真正的光路折射，用两个近似手段叠加：
 *    backdrop-filter: blur() 给空间感 + 边缘高光给厚度感。
 *    刻意**不用** filter: url(#displacement) —— 那在 ComfyUI 的
 *    litegraph 画布（本身有 transform 缩放）里会产生错位，得不偿失。
 *
 *  · **背景自适应**：玻璃颜色不能写死。底下用半透明黑 + saturate() 吸背景色，
 *    再加 brightness() 自适应亮度 —— 深色缩略图自动变暗，浅色自动提亮，
 *    这就是 Apple 说的 "颜色会受到背后内容影响，而不是一块固定的半透明灰色"。
 *
 * ---- 圆角升级 ----
 * 第 16 轮是 5px（偏锐利、像老式桌面软件）。Liquid Glass 是 iOS 26 / macOS Tahoe
 * 的语言，圆角明显更大更松弛。卡片从 5px → **10px**；内部按钮是正圆不受影响。
 * 圆角变大后 overflow: hidden 的裁切必须继续成立，否则图片四角会露出来。
 */
.h3d-refs { display: grid; gap: 6px;
            grid-template-columns: repeat(auto-fill, minmax(93px, 1fr));
            min-width: 0; }
.h3d-ref { position: relative; aspect-ratio: 1 / 1; border-radius: var(--h3d-r-md);
           overflow: hidden; background: #1a1a1a;
           border: 1px solid #2e2e2e;
           transition: border-color var(--h3d-dur-fast), box-shadow var(--h3d-dur); }
.h3d-ref:hover { border-color: #4a4a4a; }
.h3d-ref.is-empty { border-style: dashed; border-color: #333; }
.h3d-ref img { width: 100%; height: 100%; object-fit: cover; display: block; }
.h3d-ref .ph { width: 100%; height: 100%; display: flex; align-items: center;
               justify-content: center; color: #4a4a4a; font-size: 16px; }

/* 左侧色边 = 状态条（缺图红 / 未引用黄 / 正常无）。
 * 放在最底层（z-index 1），hover 遮罩盖上来时它仍然可见 —— 状态是
 * "这张图有没有问题"的常驻信息，不该被 hover 面板吞掉。 */
.h3d-ref::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0;
                   width: 4px; z-index: 1; pointer-events: none;
                   background: transparent; }
.h3d-ref.is-unused::before { background: rgba(232,184,58,.9); }   /* 未引用 → 黄 */
.h3d-ref.is-missing::before { background: rgba(224,82,82,.9); }   /* 缺图 → 红 */
.h3d-ref.is-missing { border-style: dashed; border-color: #e05252; }

/* ★ specular rim 边缘高光（Liquid Glass 招牌）
 * 用 inset box-shadow 画一道 1px 亮线贴在卡片内边。之所以用 box-shadow 而不是
 * border，是因为它能**只在内侧**、且能和状态色边（::before）共存不打架。
 * 上边亮（模拟光从上方来）、下边暗，这就是玻璃的"厚度感"。
 * z-index 2：压在图片之上、遮罩之下 —— 遮罩浮现时高光被玻璃自己接管。 */
.h3d-ref::after { content: ""; position: absolute; inset: 0; z-index: 2;
                  border-radius: inherit; pointer-events: none;
                  box-shadow: var(--h3d-rim); }
.h3d-ref:hover::after { box-shadow: inset 0 1px 0 rgba(255,255,255,.22),
                                    inset 0 -1px 0 rgba(0,0,0,.32); }

/* 分类圆点 = 左下角 8px 色点。**默认态唯一的徽标**，颜色即分类
 * （角色绿 / 环境蓝 / 道具紫 / 自行判断灰），手动指定 = 外加白圈。
 * hover 时整卡进遮罩，圆点隐去（分类大字会顶上来，不必重复）。 */
.h3d-ref .dot { position: absolute; left: 8px; bottom: 8px; z-index: 2;
                width: 9px; height: 9px; border-radius: 50%;
                background: transparent; cursor: help;
                box-shadow: 0 0 0 1px rgba(0,0,0,.45);
                transition: opacity .12s; }
.h3d-ref .dot.k-char  { background: var(--h3d-char);  }
.h3d-ref .dot.k-prop  { background: var(--h3d-prop); }
.h3d-ref .dot.k-scene { background: var(--h3d-scene); }
.h3d-ref .dot.k-other { background: var(--h3d-other); }
.h3d-ref .dot.is-manual { box-shadow: 0 0 0 1.5px #fff, 0 0 0 2.5px rgba(0,0,0,.6); }
.h3d-ref:hover .dot { opacity: 0; }        /* 进遮罩后让位给分类大字 */

/* ---- 遮罩层 .veil：唯一的一层 Liquid Glass（regular 变体）----
 * 玻璃 = 模糊(blur) + 亮度自适应(brightness) + 饱和度吸收(saturate)
 *        + Apple 的 35% 变暗层 + 一道边缘高光。
 * 全部叠在同一个 backdrop-filter 里 —— **只做一层**，不嵌套。 */
.h3d-ref .veil { position: absolute; inset: 0; z-index: 3;
                 display: flex; flex-direction: column;
                 padding: 6px 7px;
                 border-radius: inherit;
                 /* ③ 35% 深色变暗层（Apple 文档唯一的硬性数字）*/
                 background: var(--h3d-glass-dim);
                 /* ② regular 变体：模糊 + 调亮度，保证文字可读。参数走令牌 */
                 backdrop-filter: blur(var(--h3d-glass-blur))
                                  brightness(.92)
                                  saturate(var(--h3d-glass-sat));
                 -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                          brightness(.92)
                                          saturate(var(--h3d-glass-sat));
                 /* specular rim：玻璃自己的边缘高光，比卡片那层更亮 */
                 box-shadow: var(--h3d-rim-lg);
                 opacity: 0; pointer-events: none;
                 /* 出现用 Apple 偏爱的平滑曲线，不是生硬的 linear */
                 transition: opacity var(--h3d-dur) var(--h3d-ease); }
.h3d-ref:hover .veil { opacity: 1; pointer-events: auto; }
.h3d-ref.is-selecting .veil { display: none; }   /* 多选模式：遮罩让位给勾选 */

/* ① 顶部细行：左=引用次数，右=槽位号。字号压到 9.5px，不抢主信息。
 * 玻璃上的文字靠"vibrancy"等效手段保证可读：略高的字重 + 极淡的暗色投影
 * （投影只为了让文字从玻璃的亮度变化里浮出来，不是装饰）。 */
.h3d-ref .veil .vtop { display: flex; align-items: center;
                       justify-content: space-between;
                       font-size: 9.5px; line-height: 1.5;
                       color: rgba(255,255,255,.62);
                       font-variant-numeric: tabular-nums;
                       text-shadow: 0 1px 2px rgba(0,0,0,.5); }
.h3d-ref .veil .vtop .cnt { color: rgba(255,255,255,.92); font-weight: 600; }
.h3d-ref .veil .vtop .cnt.is-warn { color: #f5c542; }
.h3d-ref .veil .vtop .cnt.is-bad  { color: #ff8b8b; }
.h3d-ref .veil .vtop .sid { color: rgba(255,255,255,.5); }

/* ② 中间：分类大字（自动占满剩余高度并居中）——主信息，最醒目。
 * 玻璃上做大字要更亮更实：字号 14px、字重 700、双层 text-shadow
 * （近处一道深色压边 + 远处一道扩散）——这是让文字"贴在玻璃表面"的关键。 */
.h3d-ref .veil .vkind { flex: 1 1 auto;
                        display: flex; align-items: center; justify-content: center;
                        font-size: 14px; font-weight: 700; line-height: 1;
                        letter-spacing: .3px; color: #fff;
                        text-shadow: 0 1px 2px rgba(0,0,0,.55),
                                     0 0 8px rgba(0,0,0,.35); }
.h3d-ref .veil .vkind.k-char  { color: #6affa0; }
.h3d-ref .veil .vkind.k-prop  { color: #e2a2f7; }
.h3d-ref .veil .vkind.k-scene { color: #a3c9f5; }
.h3d-ref .veil .vkind.k-other { color: #dcdcdc; }
/* 判不出来 → 提示可手动指定，用暗灰小字，别冒充一个分类 */
.h3d-ref .veil .vkind.is-none { font-size: 10px; font-weight: 400;
                                color: rgba(255,255,255,.55); letter-spacing: 0;
                                text-shadow: 0 1px 2px rgba(0,0,0,.5); }

/* ③ 底部：三个圆形操作按钮横排。
 * ★ 宽度预算（第 16 轮实测）：卡片最小宽 93px，去 2px 边框 + veil 左右各 7px 内边距
 *   → 可用 77px。按钮 22px + 间距 4px = 3×22 + 2×4 = 74px，**留 3px 余量**
 *   防止亚像素取整 / 滚动条挤占导致换行或裁切。
 *   最初写的 30px + 8px 间距 = 106px 会横向溢出（被 verify_veil_geometry.js 抓到），
 *   后来 22px + 5px = 76px 只剩 1px 余量、太贴边 —— 这就是"设计稿好看、
 *   真机溢出"的典型坑：数字必须按**最窄卡**算，且要留缓冲，不能顶格。
 * 22px 命中区与第 15 轮侧边钮持平，但三枚集中成排、整卡遮罩可点，
 * 指针不必离开卡片就能命中，实际点击难度低得多。
 *
 * ★ 规则④：按钮**不再各带一层玻璃**（避免 glass-on-glass）。
 *   改用「半透明白填充 + 高光描边 + 顶部亮线」在玻璃面上"压印"出立体感 ——
 *   视觉上仍是同一块玻璃的一部分，但没有第二层 backdrop-filter。
 *   这也解决了浅色缩略图上的可读性：玻璃已经统一压暗 + 调亮度了，
 *   按钮只需在玻璃之上再区分一层即可。 */
.h3d-ref .veil .vacts { flex: none;
                        display: flex; align-items: center; justify-content: center;
                        gap: 4px; }
.h3d-ref .veil .vact { width: 22px; height: 22px; border-radius: 50%;
                       cursor: pointer; flex: none;
                       display: flex; align-items: center; justify-content: center;
                       /* interactive 玻璃的按压感：半透明白填充 + 顶部亮线 */
                       background: rgba(255,255,255,.16);
                       border: 1px solid rgba(255,255,255,.30);
                       box-shadow: inset 0 1px 0 rgba(255,255,255,.28),
                                   0 1px 2px rgba(0,0,0,.28);
                       color: #fff; font-size: 11px; line-height: 1;
                       text-shadow: 0 1px 2px rgba(0,0,0,.45);
                       transition: background .12s, border-color .12s,
                                   box-shadow .12s, transform .1s; }
/* hover = 玻璃被"点亮"：填充变亮、高光变强、轻微上浮 */
.h3d-ref .veil .vact:hover { background: rgba(255,255,255,.30);
                             border-color: rgba(255,255,255,.55);
                             box-shadow: inset 0 1px 0 rgba(255,255,255,.45),
                                         0 2px 5px rgba(0,0,0,.35);
                             transform: translateY(-1px); }
/* active = 按下去，回弹到原位，符合 macOS 物理直觉 */
.h3d-ref .veil .vact:active { transform: translateY(0) scale(.94);
                              box-shadow: inset 0 1px 2px rgba(0,0,0,.3); }
/* 三枚各自的语义色（都用同一个"玻璃 tint"手法：半透明色填充，不换材质） */
.h3d-ref .veil .vact.is-cycle:hover  { background: rgba(55,138,221,.55);
                                       border-color: rgba(163,201,245,.75); }
.h3d-ref .veil .vact.is-danger:hover { background: rgba(224,75,58,.55);
                                       border-color: rgba(255,139,139,.75); }

/* 圆点轻量反馈：刚被点过的卡片，圆点外圈 0.55s 出现一道细色环后收回。
 * 不靠 hover 永久保持（用户反馈：点完回头看画面，啥痕迹都没）。 */
.h3d-ref .dot::after { content: ""; position: absolute;
                       left: 50%; top: 50%;
                       width: 9px; height: 9px; border-radius: 50%;
                       transform: translate(-50%, -50%) scale(1);
                       border: 1.5px solid currentColor; opacity: 0;
                       color: transparent; }
.h3d-ref.is-just-clicked .dot::after {
  animation: h3d-dot-pulse .55s ease-out;
  color: var(--h3d-dot-ring, #4fff8f);   /* 默认绿，JS 会按分类色覆写 */
}
@keyframes h3d-dot-pulse {
  0%   { opacity: 0; transform: translate(-50%, -50%) scale(1); }
  20%  { opacity: .95; }
  100% { opacity: 0; transform: translate(-50%, -50%) scale(2.6); }
}

/* 卡片底部的「图片N」独立条已取消：槽位号挪进遮罩层右上角，
 * 常驻态只留圆点 —— 零遮挡看全图。 */

/* ---- 无障碍降级三件套（Apple 要求"不是补丁，而是默认约束"）----
 * Liquid Glass 会自动尊重系统无障碍设置。CSS 侧的等效实现：
 * ★ 第 18 轮起作用域提到**全包**（原来只在 .h3d-ref 上）——
 *   现在每个浮层都是玻璃（veil / pveil / ctx / mention / pop / bulkbar），
 *   无障碍约束必须一起跟上，否则"参考图正常、右键菜单糊成一片"。 */

/* ① Reduce Motion：关掉所有过渡/动画，玻璃变静态（不 morph、不脉冲） */
@media (prefers-reduced-motion: reduce) {
  .h3d, .h3d *, .h3d *::before, .h3d *::after,
  .h3d-ctx, .h3d-ctx *, .h3d-mention, .h3d-mention *,
  .h3d-pop, .h3d-pop * {
    animation-duration: .001ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }
  .h3d-ref .veil .vact:hover, .h3d-ref .veil .vact:active,
  .h3d-plan-card .pveil .pact:hover,
  .h3d-plan-card .pveil .pact:active,
  .h3d-lang-tab:hover, .h3d-lang-tab:active,
  .h3d-seg:hover, .h3d-bulkbar .pill:hover,
  .h3d-bulkbar .pill:active, .h3d-hd .h3d-ix:hover,
  .h3d-btn:hover, .h3d-btn:active { transform: none; }
}

/* ② Reduce Transparency：玻璃转成不透明磨砂 —— 保证阅读对比度。
 *    Apple 原文："开启后，玻璃会更偏磨砂、更不透明，以保证阅读对比度。"
 *    做法：把 backdrop-filter 关掉、背景改成接近实色的深色。
 *    ★ 覆盖全部浮层：漏一个就会在开启该设置时"一半实一半透"。 */
@media (prefers-reduced-transparency: reduce) {
  .h3d-ref .veil,
  .h3d-plan-card .h3d-plan-bd.is-preview .pveil,
  .h3d-ctx, .h3d-mention, .h3d-pop, .h3d-bulkbar,
  .h3d-glass, .h3d-glass--strong {
    background: rgba(18, 18, 20, .96);
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
  }
  .h3d-ref::after { box-shadow: none; }
  /* 压印控件（按钮/胶囊）本身没 blur，但半透明白底在全不透明模式下
     会显得"脏"—— 提高不透明度让它读起来像实体按钮。 */
  .h3d-btn, .h3d-lang-tab, .h3d-bulkbar .pill,
  .h3d-ref .veil .vact, .h3d-plan-card .pveil .pact {
    background: rgba(255,255,255,.20);
    border-color: rgba(255,255,255,.34);
  }
}

/* ③ Increase Contrast：整体更高对比、边界更清晰。
 *    做法：加深玻璃底色、加强高光描边、文字提到纯白、圆点加白圈。 */
@media (prefers-contrast: more) {
  .h3d-ref .veil,
  .h3d-plan-card .h3d-plan-bd.is-preview .pveil,
  .h3d-ctx, .h3d-mention, .h3d-pop, .h3d-bulkbar {
    background: rgba(0, 0, 0, .78);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,.5);
  }
  .h3d-ref .veil .vtop,
  .h3d-plan-card .pveil .ptop { color: rgba(255,255,255,.9); }
  .h3d-ref .veil .vtop .cnt,
  .h3d-ref .veil .vkind,
  .h3d-plan-card .pveil .pkind,
  .h3d-plan-card .pveil .pkind .sub,
  .h3d-ctx-item, .h3d-mention-item { color: #fff; }
  .h3d-ref .veil .vact,
  .h3d-plan-card .pveil .pact,
  .h3d-btn, .h3d-lang-tab, .h3d-bulkbar .pill {
    background: rgba(255,255,255,.28);
    border-color: rgba(255,255,255,.7); }
  .h3d-ref .dot { box-shadow: 0 0 0 1.5px #fff, 0 0 0 2.5px rgba(0,0,0,.8); }
  .h3d-ctx-sep { background: rgba(255,255,255,.34); }
}

/* 多选模式 = 卡片四周描边变绿，浮出左上角勾选圈
 * 多选模式下点击 = 选中/取消选中，不再是切分类 */
.h3d-ref.is-selecting { cursor: pointer; }
.h3d-ref.is-selected { border-color: #4fff8f; box-shadow: inset 0 0 0 1.5px #4fff8f; }
.h3d-ref .pick { display: none;
                 position: absolute; top: 4px; left: 4px; z-index: 4;
                 width: 18px; height: 18px; border-radius: 50%;
                 background: rgba(0,0,0,.78); border: 1.5px solid #aaa;
                 color: #fff; font-size: 12px; line-height: 1;
                 align-items: center; justify-content: center; }
.h3d-ref.is-selecting .pick { display: flex; }
.h3d-ref.is-selected .pick { background: #4fff8f; border-color: #4fff8f; color: #0d1a11; }
.h3d-ref.is-selected .pick::after { content: "✓"; font-weight: 700; }

/* 「添加图片」占位卡片（第 17 轮：圆角跟卡片一起升到 10px，
 * hover 时给一道 glass 高光，和参考图卡片的 specular rim 呼应） */
.h3d-ref-add { position: relative;
               aspect-ratio: 1 / 1; border-radius: 10px; cursor: pointer;
               border: 1px dashed #3a3a3a; background: #171717;
               display: flex; flex-direction: column; align-items: center;
               justify-content: center; gap: 2px; color: #6f6f6f; font-size: 9.5px;
               transition: border-color .12s, color .12s, background .12s; }
.h3d-ref-add::after { content: ""; position: absolute; inset: 0;
                      border-radius: inherit; pointer-events: none;
                      box-shadow: inset 0 1px 0 rgba(255,255,255,.14);
                      transition: box-shadow .12s; }
.h3d-ref-add:hover { border-color: #4fff8f; color: #4fff8f; background: #1a1f1a; }
.h3d-ref-add:hover::after { box-shadow: inset 0 1px 0 rgba(79,255,143,.35); }
.h3d-ref-add .plus { font-size: 17px; line-height: 1; }
.h3d-file { display: none; }

/* 顶部「参考图 N / M」+ 缺失警示条 */
.h3d-hd .h3d-refstat { display: inline-flex; align-items: center; gap: 6px;
                        font-size: 10px; color: #8a8a8a; margin-left: 2px; }
.h3d-hd .h3d-refstat .num { color: #d8d8d8; font-weight: 600;
                            font-variant-numeric: tabular-nums; }
.h3d-hd .h3d-refstat .sep { color: #555; }
.h3d-hd .h3d-refstat .miss { padding: 1px 5px; border-radius: 3px;
                              background: rgba(232,184,58,.18); color: #e8b83a;
                              font-size: 9.5px; line-height: 1.4; }
/* 「已齐」用暗绿，跟「缺 N」的黄形成对比；不是错误态就不该抢视线 */
.h3d-hd .h3d-refstat .miss.is-ok { background: rgba(79,255,143,.14); color: #4fff8f; }

/* 批量应用条：进入多选 + 至少选一张时，从面板底部滑出。
 * 第 17 轮：这是**第二处浮在内容之上的功能层**，按规则①也算 Liquid Glass 的适用范围
 * （它是"控件层"，不是内容层）。所以换成玻璃材质，但用**偏绿 tint** 表达"应用态"。
 * ★ 关键是几何：这是浮动条，必须离边缘留白（Apple 的 floating 控件从不贴边），
 *   所以加 margin 而不是贴死底部。 */
/* 批量操作条：玻璃浮层 + 绿色 tint（表示"多选态"）。
   ★ 第 18 轮起改吃令牌 —— 与 .h3d-ctx / .h3d-pop / .veil 完全同源，
     改 --h3d-glass-blur 一处，全包玻璃一起变。 */
.h3d-bulkbar { position: sticky; bottom: 0; z-index: 5;
               display: flex; align-items: center; gap: 8px;
               margin: 6px 2px 2px; padding: 7px 10px;
               border-radius: var(--h3d-r-lg);
               background: rgba(20, 40, 28, .50);
               backdrop-filter: blur(var(--h3d-glass-blur))
                                brightness(var(--h3d-glass-bright))
                                saturate(var(--h3d-glass-sat));
               -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                        brightness(var(--h3d-glass-bright))
                                        saturate(var(--h3d-glass-sat));
               border: 1px solid rgba(79, 255, 143, .34);
               box-shadow: var(--h3d-rim-lg), 0 6px 18px rgba(0,0,0,.45);
               color: #e2f7e7; font-size: 11px;
               transition: background var(--h3d-dur-fast),
                           box-shadow var(--h3d-dur-fast); }
.h3d-bulkbar .lbl { color: #9fd8ae; }
.h3d-bulkbar .pill { padding: 4px 11px; border-radius: var(--h3d-r-pill);
                     cursor: pointer;
                     background: var(--h3d-press-bg);
                     border: 1px solid var(--h3d-press-bd);
                     box-shadow: var(--h3d-rim-hi);
                     color: #eaf7ec; font-size: 10.5px; line-height: 1.5;
                     transition: background var(--h3d-dur-fast),
                                 box-shadow var(--h3d-dur-fast), transform .1s; }
.h3d-bulkbar .pill:hover { background: var(--h3d-press-bg-hi);
                           border-color: var(--h3d-press-bd-hi);
                           box-shadow: inset 0 1px 0 rgba(255,255,255,.36),
                                       var(--h3d-lift-sm);
                           transform: translateY(-1px); }
.h3d-bulkbar .pill:active { transform: translateY(0) scale(.95); }
/* 四类分类胶囊：同一个玻璃 tint 手法，只换色相 —— 色值走 --h3d-* 令牌，
   与参考图卡片的分类圆点/大字**永远同色**。 */
.h3d-bulkbar .pill.k-char:hover  { background: var(--h3d-char-a);  border-color: var(--h3d-char);  color: #fff; }
.h3d-bulkbar .pill.k-prop:hover  { background: var(--h3d-prop-a);  border-color: var(--h3d-prop);  color: #fff; }
.h3d-bulkbar .pill.k-scene:hover { background: var(--h3d-scene-a); border-color: var(--h3d-scene); color: #fff; }
.h3d-bulkbar .pill.k-other:hover { background: var(--h3d-other-a); border-color: var(--h3d-other); color: #fff; }
.h3d-bulkbar .cancel { margin-left: auto; cursor: pointer;
                       color: rgba(255,255,255,.6);
                       font-size: 14px; line-height: 1; padding: 0 4px;
                       transition: color .12s; }
.h3d-bulkbar .cancel:hover { color: #fff; }

.h3d-hint { font-size: 10px; color: #6f6f6f; line-height: 1.45; }

/* ========= 3) 分镜时间线 ========= */
.h3d-bar-row { display: flex; align-items: center; gap: 9px; min-width: 0; }
.h3d-bar { flex: 1 1 auto; height: 5px; background: #2a2a2a;
           border-radius: 3px; overflow: hidden; min-width: 40px; }
.h3d-bar > i { display: block; height: 100%; width: 0;
               background: linear-gradient(90deg, #2a6b4a, #4fff8f);
               border-radius: 3px; transition: width .15s ease; }
.h3d-bar-text { font-size: 10px; color: #8a8a8a; flex: none;
                font-variant-numeric: tabular-nums; }

/* 段块：未渲=静态缩略图；渲完=实时视频替换
   ★ Liquid Glass 第 1 条：玻璃**不上内容层** —— 段块里的缩略图/视频是内容，
     所以只把"外框"做成玻璃语言（圆角 + rim 高光 + 状态描边），
     绝不给缩略图本身套 backdrop-filter。 */
.h3d-segs { display: flex; gap: 3px; min-width: 0; }
.h3d-seg { position: relative; flex: 1 1 0; min-width: 30px; height: 87px;
           /* ★ 与 .h3d-ref 同款圆角：原本段块用 --h3d-r-sm（4px）偏方正，
              参考图用 --h3d-r-md（8px）更柔和。两个区域视觉语言统一一下。 */
           border-radius: var(--h3d-r-md); overflow: hidden;
           background: rgba(255,255,255,.04);
           border: 1px solid rgba(255,255,255,.10);
           box-shadow: var(--h3d-rim);
           cursor: pointer;
           transition: border-color var(--h3d-dur-fast),
                       box-shadow var(--h3d-dur-fast),
                       transform var(--h3d-dur-fast); }
.h3d-seg:hover { border-color: rgba(255,255,255,.24);
                 box-shadow: var(--h3d-rim), var(--h3d-lift-sm);
                 transform: translateY(-1px); }
/* ★ 与 .h3d-ref .dot 同款：渲染状态点用参考图同色令牌（--h3d-acc / --h3d-bad /
   灰中点），统一两个区域的状态语言。 */
.h3d-seg.is-done   { border-color: rgba(79,255,143,.42); }
.h3d-seg.is-active { border-color: var(--h3d-acc);
                     box-shadow: var(--h3d-rim), 0 0 0 1px rgba(79,255,143,.35),
                                 0 0 12px rgba(79,255,143,.22); }
.h3d-seg video, .h3d-seg img { width: 100%; height: 100%; object-fit: cover;
                               display: block; }
.h3d-seg .blank { width: 100%; height: 100%; opacity: .5;
                  background: repeating-linear-gradient(45deg,
                    transparent 0 5px, rgba(255,255,255,.045) 5px 6px); }
.h3d-seg .no { position: absolute; top: 2px; left: 4px; z-index: 2;
               font-size: 9.5px; font-weight: 700; color: #fff;
               text-shadow: 0 1px 3px rgba(0,0,0,.95); }
/* 段块右上"绿勾 ✓" —— 用户第 23 轮明确要求保留不动，沿用旧写法 */
.h3d-seg .st { position: absolute; top: 2px; right: 4px; z-index: 2;
               font-size: 9.5px; line-height: 1; }
.h3d-seg .st.is-done   { color: #4fff8f; }
.h3d-seg .st.is-todo   { color: #6a6a6a; }
.h3d-seg .st.is-active { color: #4fff8f; animation: h3d-blink .9s infinite; }
@keyframes h3d-blink { 50% { opacity: .25; } }
/* ★ 第 23 轮新增：参考图同款圆形状态点。
   位置：缩略图右下角（避开 .no 段号左上 / .st 状态右上 / .dur 时长右下=其实会撞，
   但参考图也是 dot 在左下，所以这里换到右下跟时长并排更对称；时长在 bottom:16px，
   dot 放 bottom:4px 贴底）。
   语义：渲染状态（done=实心绿 / todo=灰 / active=蓝闪烁）—— 与 .st 字符绿勾并存，
   绿勾是"字符徽标"，dot 是"参考图同款状态色块"。 */
.h3d-seg .dot { position: absolute; left: 6px; bottom: 6px; z-index: 2;
                width: 9px; height: 9px; border-radius: 50%;
                background: transparent;
                box-shadow: 0 0 0 1px rgba(0,0,0,.55);
                transition: opacity .12s; }
.h3d-seg .dot.is-done   { background: var(--h3d-acc); }
.h3d-seg .dot.is-todo   { background: #6a6a6a; }
.h3d-seg .dot.is-active { background: #4aa9ff; animation: h3d-blink .9s infinite; }
.h3d-seg:hover .dot { opacity: .9; }     /* 参考图:hover .dot 是 opacity:0，这里
                                            改成 .9 让 hover 时也可见（段块没有遮罩）。 */
/* 「提示词已改、待重渲」角标：段块右上琥珀色 ✎（让开状态 ●/✓/○ 的 right:4px），
   段块加一圈琥珀描边，一眼看出哪几段改过、需要重渲。 */
.h3d-seg .dirty { position: absolute; top: 2px; right: 14px; z-index: 3;
                  font-size: 9.5px; line-height: 1; color: #ffcf6b;
                  text-shadow: 0 1px 3px rgba(0,0,0,.95); }
.h3d-seg.is-dirty { box-shadow: 0 0 0 1px #ffcf6b99 inset; }

/* ★ 第 28 轮：段块「重渲」按钮改成参考图同款 Liquid Glass 浮窗 + 圆形按钮。
   之前是底部横条（"↻ 重渲该段"），占满段宽、跟视频画面抢戏；
   现在缩成右下角玻璃面板，里面一个 22px 圆形 ↻ 按钮 —— 与 .h3d-ref .vact
   完全同源（背景 + 模糊 + 高光都用同一组 --h3d-* 令牌），视觉语言统一。
   段块太矮放不下参考图的"顶部细行 + 中间大字 + 底部按钮"三段式，
   所以**只保留圆形按钮**这个核心视觉元素，玻璃材质照搬。 */
.h3d-seg .re-glass { position: absolute; inset: 0; z-index: 4;
                     padding: 5px;
                     /* ★ 第 39 轮：勾选框搬走后玻璃里只剩 ↻，回到第 32 轮
                        「单按钮居中」形态。hover 显现后只露 ↻ 入口。 */
                     display: flex; align-items: center; justify-content: center;
                     border-radius: inherit;
                     /* ★ 第 42 轮（2026-09-17）：这一层铺在**视频缩略图**上面，
                        原样照搬玻璃的模糊滤镜会把画面糊成一团（12px 高斯 +
                        35% 暗 —— 用户实测「hover 时画面一团模糊」，第 29 轮
                        记录里就写了这个取舍，现在按用户要求翻过来）。
                        改成「只压暗、不模糊」：薄纱 22% 黑，画面照常清晰可辨；
                        ↻ 按钮自己有半透明白底 + 高光描边 + 顶部亮线，照样读得出。
                        覆盖范围仍是 inset:0（第 29 轮要求：铺满整个缩略图）。
                        ★ 本块内不得再出现任何模糊滤镜 —— 见回归断言。 */
                     background: rgba(0, 0, 0, .22);
                     box-shadow: var(--h3d-rim-lg);
                     opacity: 0; pointer-events: none;
                     transition: opacity var(--h3d-dur) var(--h3d-ease); }
.h3d-seg:hover .re-glass { opacity: 1; pointer-events: auto; }
/* 复用参考图 .h3d-ref .veil .vact 的圆形按钮写法 —— 半透明白填充 + 高光描边
   + 顶部亮线，绝不再叠一层 backdrop-filter（玻璃上盖玻璃 = 噪点叠叠乐）。
   ★ 第 32 轮：圆形按钮位置由 .re-glass 的 flex 推到中心（与本卡片
     「重渲该段」按钮 + 参考图卡片的「↻」圆形按钮三处保持一致 —— 同语义
     的入口在同一视觉位置，鼠标记一次就够）。 */
.h3d-seg .re-glass .vact { width: 28px; height: 28px; border-radius: 50%;
                            cursor: pointer; flex: none;
                            /* ★ 第 33 轮：z-index: 6 只在 .re-glass 自己的
                               stacking context 内生效 —— 它带
                               position:absolute + z-index:4，本身就是独立
                               context，所以 .vact 实际最高到 z:4，
                               不会真的覆盖 z:5 的 .run。
                               视觉上 .run 与 .vact 横向错开 22px，本来就不重叠，
                               这个 z-index 留着是为了将来 .re-glass 内再添别的
                               子元素时（如「✗ 取消重渲」）.vact 排在最上。
                               ★ 第 42 轮：模糊滤镜已从 .re-glass 移除
                               （媒体层不许糊），stacking context 改由
                               position:absolute + z-index 提供，结论不变。 */
                            z-index: 6;
                            display: flex; align-items: center; justify-content: center;
                            background: rgba(255,255,255,.16);
                            border: 1px solid rgba(255,255,255,.30);
                            box-shadow: inset 0 1px 0 rgba(255,255,255,.28),
                                        0 1px 2px rgba(0,0,0,.28);
                            color: #fff; font-size: 14px; line-height: 1;
                            text-shadow: 0 1px 2px rgba(0,0,0,.45);
                            transition: background .12s, border-color .12s,
                                        box-shadow .12s, transform .1s; }
.h3d-seg .re-glass .vact:hover { background: rgba(255,255,255,.30);
                                  border-color: rgba(255,255,255,.55);
                                  box-shadow: inset 0 1px 0 rgba(255,255,255,.45),
                                              0 2px 5px rgba(0,0,0,.35);
                                  transform: translateY(-1px); }
.h3d-seg .re-glass .vact:active { transform: translateY(0) scale(.94);
                                   box-shadow: inset 0 1px 2px rgba(0,0,0,.3); }
/* ★ 第 29 轮：玻璃铺满（z-index 4）后，下层角标默认 z-index 2-3 会被盖住。
   把所有角标提到 z-index 5 浮在玻璃之上 —— 段号、状态、时长等关键信息
   hover 时仍清晰可读。视频/图片 z-index 保持默认（被玻璃柔和调暗），
   这是参考图的视觉语言：hover 看段信息 + 操作入口，不看视频细节。 */
.h3d-seg .no, .h3d-seg .st, .h3d-seg .dot,
.h3d-seg .dirty, .h3d-seg .dur, .h3d-seg .run { z-index: 5; }
/* ★ 第 39 轮：勾选框常驻在段块左下角（参考 ComfyUI_MiniMaxH3_Director 的
   canvas 自绘勾选框位置）—— 上一轮装进 .re-glass，必须 hover 段块才能看见/点，
   用户原话「同时勾选多个不同分镜缩略图」需要一眼就看见全部勾选框。
   改为：绝对定位左下角 4px（与 .h3d-seg 的 .dot 同侧但不重叠 —— .dot 在右下、
   .run 在左下，互不挤占）。
   集合语义、aria-checked / data-run="on" 触发绿底白勾与上轮一致。 */
.h3d-seg .run {
              position: absolute; left: 4px; bottom: 4px;
              width: 20px; height: 20px; margin: 0; padding: 0;
              flex: none; cursor: pointer;
              border: 1px solid rgba(255,255,255,.55);
              border-radius: 3px;
              background: rgba(0,0,0,.55);
              color: #fff;
              font-size: 13px; font-weight: 700; line-height: 1;
              display: flex; align-items: center; justify-content: center;
              transition: background .12s, border-color .12s, color .12s;
              backdrop-filter: blur(2px); }
.h3d-seg .run:hover { background: rgba(255,255,255,.18);
                border-color: rgba(255,255,255,.85); color: #fff; }
/* 勾选态：绿底白勾（data-run="on" 触发）—— 用户明确要求还原成这个可勾选的小框 */
.h3d-seg .run[data-run="on"] {
                background: #4fff8f; border-color: #4fff8f;
                color: #0d1a11; }
.h3d-seg .run[data-run="on"]::before { content: "✓"; font-weight: 700; }
.h3d-seg.is-run { box-shadow: inset 0 0 0 1px rgba(79,255,143,.5); }

/* 本段没有解析到提示词正文（数据缺口，不是显示故障）：红边提示，与
   .is-dirty（提示词已改待重渲）的琥珀色区分开。 */
.h3d-seg.is-noprompt { border-color: #7a3a3a; box-shadow: inset 0 0 0 1px rgba(160,60,60,.35); }

/* 借鉴融合：段块右键菜单 + 时间线键盘焦点（参考 AIMixer 的 contextmenu/keydown 交互）。
   菜单挂 body 下用 fixed 定位，避免被面板的 overflow / transform 裁掉。 */
/* 右键菜单：这是**悬浮功能层**，Liquid Glass 最正统的用武之地
   （Apple 自己的菜单就是玻璃）。用 strong 底色 —— 菜单里字多，要更实。 */
.h3d-ctx { position: fixed; z-index: 9999; min-width: 138px; padding: 5px;
           border-radius: var(--h3d-r-lg);
           background: var(--h3d-glass-bg-strong);
           backdrop-filter: blur(var(--h3d-glass-blur))
                            brightness(var(--h3d-glass-bright))
                            saturate(var(--h3d-glass-sat));
           -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                    brightness(var(--h3d-glass-bright))
                                    saturate(var(--h3d-glass-sat));
           border: 1px solid rgba(255,255,255,.12);
           box-shadow: var(--h3d-rim-lg), 0 10px 30px rgba(0,0,0,.55);
           font-size: 11px;
           transform-origin: top left;
           animation: h3d-ctx-in .13s var(--h3d-ease); }
@keyframes h3d-ctx-in {
  from { opacity: 0; transform: scale(.96) translateY(-2px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}
.h3d-ctx-item { padding: 6px 9px; border-radius: var(--h3d-r-xs); color: #e4e4ea;
                cursor: pointer; white-space: nowrap;
                transition: background var(--h3d-dur-fast), color var(--h3d-dur-fast); }
.h3d-ctx-item:hover { background: var(--h3d-press-bg-hi); color: var(--h3d-acc); }
.h3d-ctx-sep { height: 1px; margin: 4px 2px;
               background: linear-gradient(90deg, transparent,
                           rgba(255,255,255,.14) 18%,
                           rgba(255,255,255,.14) 82%, transparent); }
.h3d-segs { outline: none; }
.h3d-segs:focus-visible { outline: 1px solid #4fff8f; outline-offset: 2px; }

/* 提示词 <Picture N> / <Video K> / <Audio J> 胶囊（B1，只读，可点定位参考图） */
.h3d-tok { display: inline-flex; align-items: center; gap: 4px; height: 16px;
           padding: 0 5px 0 2px; border-radius: 3px; vertical-align: middle;
           background: #243024; border: 1px solid #2f6b45; color: #9fe8b8;
           font-size: 10px; white-space: nowrap; cursor: pointer; }
.h3d-tok img { width: 12px; height: 12px; border-radius: 2px; object-fit: cover; }
.h3d-tok.is-missing { background: #2a2222; border-color: #6b3a3a; color: #e8a0a0; }
.h3d-tok:hover { border-color: #4fff8f; color: #fff; }

/* @ 提及菜单（B2）：挂在 body 下、position: fixed，避免被面板 overflow 裁切。
   与 .h3d-ctx 同为悬浮功能层 → 同一套 Liquid Glass 材质（strong 变体）。 */
.h3d-mention { position: fixed; z-index: 9999; max-height: 224px; overflow-y: auto;
               background: var(--h3d-glass-bg-strong);
               backdrop-filter: blur(var(--h3d-glass-blur))
                                brightness(var(--h3d-glass-bright))
                                saturate(var(--h3d-glass-sat));
               -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                        brightness(var(--h3d-glass-bright))
                                        saturate(var(--h3d-glass-sat));
               border: 1px solid rgba(255,255,255,.12);
               border-radius: var(--h3d-r-lg);
               box-shadow: var(--h3d-rim-lg), 0 10px 30px rgba(0,0,0,.55);
               padding: 5px; min-width: 184px;
               transform-origin: top left;
               animation: h3d-ctx-in .13s var(--h3d-ease); }
.h3d-mention-item { display: flex; align-items: center; gap: 7px; padding: 5px 7px;
                    border-radius: var(--h3d-r-xs); cursor: pointer;
                    font-size: 10.5px; color: #e2e2e2;
                    transition: background var(--h3d-dur-fast); }
.h3d-mention-item.is-cur { background: var(--h3d-press-bg-hi); }
.h3d-mention-item img { width: 18px; height: 18px; border-radius: 3px;
                        object-fit: cover; flex: none; }
.h3d-mention-item .nm { color: #7d7d7d; margin-left: auto; font-size: 9.5px;
                        overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
                        max-width: 96px; }
.h3d-mention-item.is-missing { color: #b98a8a; }
/* 菜单底部提示：还有 N 个槽位没接图（信息告知，不参与选择） */
.h3d-mention-empty { padding: 5px 7px; margin-top: 2px; font-size: 9.5px;
                     color: #8a8a8a; border-top: 1px solid #2e2e2e;
                     line-height: 1.35; cursor: default; }

/* 提示词胶囊预览条（B2）：把 prompt_lines 里写的 <Picture N> 渲染成胶囊，
 * 手打了不存在的槽位会显示成红框胶囊 + 一行警告。只在真的写了 token 时出现。 */
.h3d-tokview { display: none; flex-wrap: wrap; align-items: center; gap: 4px;
               margin-top: 4px; padding: 4px 6px; border-radius: 5px;
               background: #141519; border: 1px solid #26282e;
               max-height: 92px; overflow-y: auto; }
.h3d-tokview.is-on { display: flex; }
.h3d-tokwarn { width: 100%; font-size: 9.5px; color: #e8a0a0; line-height: 1.4; }
.h3d-tokok { width: 100%; font-size: 9.5px; color: #6f7f74; line-height: 1.4; }
/* 参考图卡片被胶囊/提及点中时的高亮闪一下 */
.h3d-ref.is-flash { border-color: #4fff8f !important;
                    box-shadow: 0 0 0 2px rgba(79,255,143,.5); transition: none; }

/* hover 视频预览。★ 必须用 position: fixed —— 面板长在节点内部，从 DOM
 * widget 容器到画布层普遍带 overflow 裁剪，absolute 浮层会被切掉一半。
 * 坐标在 mouseenter 里按段块的实际屏幕位置算（见 render()）。 */
/* 段块 hover 弹出的视频预览：又一个悬浮功能层 → Liquid Glass strong。
   ★ 但 video 本体是内容，不进玻璃层 —— 玻璃只做"相框"。 */
.h3d-pop { position: fixed; z-index: 99999; display: none;
           transform: translate(-50%, -100%);
           background: var(--h3d-glass-bg-strong);
           backdrop-filter: blur(var(--h3d-glass-blur))
                            brightness(var(--h3d-glass-bright))
                            saturate(var(--h3d-glass-sat));
           -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                    brightness(var(--h3d-glass-bright))
                                    saturate(var(--h3d-glass-sat));
           border: 1px solid rgba(255,255,255,.13);
           border-radius: var(--h3d-r-lg); padding: 4px;
           box-shadow: var(--h3d-rim-lg), 0 14px 36px rgba(0,0,0,.68);
           pointer-events: none;
           animation: h3d-ctx-in .13s var(--h3d-ease); }
.h3d-pop.is-on { display: block; }
.h3d-pop video { display: block; width: 340px; max-height: 220px;
                 border-radius: var(--h3d-r-sm); background: #000; }
.h3d-pop .meta { font-size: 10px; color: #cfcfcf; text-align: center;
                 padding: 4px 4px 2px; line-height: 1.35; }
.h3d-pop .meta .tip { display: block; color: #7d7d7d; font-size: 9px; }

/* ── 时间标尺（A2 → 第 40 轮重写 → 第 41 轮统一到 Apple Liquid Glass）────
   ★ 第 40 轮为什么重写：上一版把 0…总时长的所有刻度**绝对定位**成一整行 <i>，
     而 .h3d-ruler 当时**连一条 CSS 都没有**（没有 position:relative /
     height / overflow），N 个 "0s 2s 4s 6s…" 全部叠在同一处 —— 用户看到
     的就是那一串"乱码"。刻度本身没错，是容器把字压到了一起。
   ★ 第 40 轮的新结构：标尺与 .h3d-segs **同构**（同样的 display:flex /
     gap:3px / 每格同样的 flexGrow = 该段真实时长），所以每一格压在它对应的
     分镜缩略图正上方；刻度用**百分比**画在格子内部，段数再多也不互相压字。
   ★ 第 41 轮两处调整（用户要求）：
     1) **取消格子里的段号** —— 段块左上角 .no 已经有段号，标尺上再来一份是
        重复信息，还白占一整行高度。段号取消后，"对应缩略图"这条仍然成立：
        靠**段分界竖线**对齐，分界线正对着缩略图之间那 3px 的缝。
     2) **标尺本体做成与 .h3d-ref 同款的玻璃条** —— 同圆角（--h3d-r-md）、
        同描边（1px #2e2e2e）、同 specular rim（::after + --h3d-rim），
        材质照抄 .h3d-glass 工具类的三条令牌（blur + brightness + saturate）。
        改令牌即全站生效，不在这里写死 rgba。 */
.h3d-ruler { position: relative; display: flex; gap: 3px;
             flex: none; height: 22px; margin: 0 0 3px; min-width: 0;
             border-radius: var(--h3d-r-md);
             border: 1px solid #2e2e2e;
             overflow: hidden;
             /* 玻璃材质 = 与 .h3d-glass 工具类同一套令牌 */
             background: var(--h3d-glass-bg);
             backdrop-filter: blur(var(--h3d-glass-blur))
                             brightness(var(--h3d-glass-bright))
                             saturate(var(--h3d-glass-sat));
             -webkit-backdrop-filter: blur(var(--h3d-glass-blur))
                                      brightness(var(--h3d-glass-bright))
                                      saturate(var(--h3d-glass-sat)); }
/* specular rim：与 .h3d-ref::after 同款手法 —— 用 inset box-shadow 而不是
   border，才能只画在内侧、且不被上面的 overflow:hidden 影响。
   上边亮、下边暗 = 光从上方来 = 玻璃的"厚度感"。
   z-index 3：压在所有刻度之上（刻度是 1/2 层），但不吃指针事件。 */
.h3d-ruler::after { content: ""; position: absolute; inset: 0; z-index: 3;
                    border-radius: inherit; pointer-events: none;
                    box-shadow: var(--h3d-rim); }
.h3d-rul-cell { position: relative; flex: 1 1 0; min-width: 0; height: 100%; }
/* 该段真实时长（new+handoff）：格子右上角，与段块右下 .dur 的"新增秒数"互补。
   段号取消后顶部那条让它独占，只在格子够宽时出现 */
.h3d-rul-len { position: absolute; right: 3px; top: 1px;
               height: 9px; line-height: 9px;
               font-size: 8px; color: rgba(255,255,255,.36);
               text-shadow: var(--h3d-txt-shadow);
               font-variant-numeric: tabular-nums; }
/* 段分界：格子左边界一根竖线，**不带数字** —— 段界是 11.0 / 23.6 / 36.2s
   这类小数，写出来会紧贴 10s / 20s 变成新的乱码。
   它唯一的任务就是"对齐下面的缩略图分界"，所以比主刻度更亮一档。 */
.h3d-rul-t0 { position: absolute; left: 0; bottom: 3px; width: 0; height: 9px;
              border-left: 1px solid rgba(255,255,255,.34); }
/* 整数秒主刻度（带数字）：间隔由 rulerStep() 保证 ≥46px，永不互相压字 */
.h3d-rul-maj { position: absolute; bottom: 3px; width: 0; height: 9px;
               border-left: 1px solid rgba(255,255,255,.24); }
.h3d-rul-maj > span { position: absolute; left: 3px; top: -1px;
                      font-size: 8.5px; line-height: 10px; color: #b9b9b9;
                      text-shadow: var(--h3d-txt-shadow);
                      white-space: nowrap; font-variant-numeric: tabular-nums; }
/* 格内细刻度：只画小竖线不写字，不参与标签排布，所以再密也不会变乱码 */
.h3d-rul-min { position: absolute; bottom: 3px; width: 0; height: 4px;
               border-left: 1px solid rgba(255,255,255,.13); }
/* 收尾：最右端的终点线 + 总时长（各段真实时长之和）。
   ★ height:100% 不能省 —— 它是 0 宽的 flex item，不撑高的话里面的
     bottom:3px 会相对一个 0 高的盒子算，数字会飘到条子外面去。 */
.h3d-rul-end { position: relative; flex: 0 0 0; width: 0; min-width: 0;
               height: 100%; }
.h3d-rul-end > i { position: absolute; right: 0; bottom: 3px; width: 0; height: 9px;
                   border-right: 1px solid rgba(255,255,255,.34); }
.h3d-rul-end > span { position: absolute; right: 3px; bottom: 3px;
                      font-size: 8.5px; line-height: 10px; color: #b9b9b9;
                      text-shadow: var(--h3d-txt-shadow);
                      white-space: nowrap; font-variant-numeric: tabular-nums; }
/* 格子窄到放不下字（段数很多时）：只留刻度线，数字全隐 */
.h3d-rul-cell.is-tiny .h3d-rul-len,
.h3d-rul-cell.is-tiny .h3d-rul-maj > span { display: none; }

/* 轨道：小马脚下这条线 —— 两层叠在一起：
 *   .base = 暗虚线（未渲染），.fill = 实线（已跑到），宽和亮度由 JS 按进度给。
 *   小马跑到哪儿，.fill 就长到哪儿：由虚到实、由暗到亮。 */
.h3d-rail { position: relative; height: 38px; margin-top: 1px;
            overflow: hidden; }
.h3d-rail-line { position: absolute; left: 0; right: 0; bottom: 5px;
                 display: flex; height: 1px; }
.h3d-rail-seg { position: relative; flex: 1 1 0; min-width: 4px; height: 1px; }
.h3d-rail-seg > .base { position: absolute; inset: 0;
  background: repeating-linear-gradient(90deg,
    #3a3a3a 0 4px, transparent 4px 7px); }
.h3d-rail-seg > .fill { position: absolute; left: 0; top: 0; height: 1px;
  width: 0; background: #4fff8f; box-shadow: none;
  transition: width .08s linear, background .08s linear; }

/* 蹄印：奔马每跑一段落一只，落下后自己淡出，不堆积 */
.h3d-hoofs { position: absolute; inset: 0; pointer-events: none; }
.h3d-hoof { position: absolute; display: block; width: 4px; height: 5px;
            border-radius: 52% 52% 42% 42% / 64% 64% 36% 36%;
            background: #8aa63f; opacity: 0;
            animation: h3d-hoof-fade 2.4s ease-out forwards; }
.h3d-hoof::after { content: ""; position: absolute; left: 50%; top: 14%;
                   width: 1px; height: 68%; margin-left: -.5px;
                   background: rgba(0,0,0,.5); }
@keyframes h3d-hoof-fade {
  0%   { opacity: 0;   transform: scale(.5); }
  14%  { opacity: .85; transform: scale(1); }
  100% { opacity: 0;   transform: scale(1.04); }
}

/* 小马：32px，单向向前跑（不回头）。SVG 原图马头朝左，翻过来朝右。
 * bottom 是 11px（不是 7px）：viewBox 换成 94 160 834 704 之后，剪影底边
 * 正好落在 viewBox 底边上（马蹄最低点 y=864），马脚贴住 div 底；要维持
 * 原来「马蹄距轨道线约 10.75px」的位置，bottom 就得跟着抬 3.75px。 */
.h3d-pony { position: absolute; bottom: 11px; left: 0;
            width: 32px; height: 27px; will-change: left; }
.h3d-pony svg { width: 100%; height: 100%; display: block;
                transform: scaleX(-1); }

/* 状态行 —— 整体去掉「黑底硬块」，改成跟其他控件同语言的 pill：
 *   - 默认：透明底 + 1px 灰描边 + 左侧 3px 灰 accent
 *   - 渲染中：accent 变绿且呼吸，边框深绿，文字浅绿
 *   - 完成：accent 实绿，边框深绿，文字柔和绿
 *   - 错误：accent 红，边框深红，文字浅红
 * 内容按 " · " 拆成 caption / shot_id 徽标 / task 绿色徽标三档。 */
.h3d-status { font-size: 10.5px; color: #9a9a9a; line-height: 1.45;
              padding: 5px 9px 5px 11px; background: transparent;
              border: 1px solid #2a2a2a; border-radius: 4px; min-height: 24px;
              display: flex; align-items: center; gap: 7px; flex-wrap: wrap;
              font-variant-numeric: tabular-nums; min-width: 0;
              overflow-wrap: anywhere; word-break: break-word; }
.h3d-status::before { content: ""; flex: none; width: 3px; height: 14px;
                      border-radius: 2px; background: #3a3a3a;
                      transition: background .15s, box-shadow .15s; }
.h3d-status .h3d-c   { color: #9a9a9a; }
.h3d-status .h3d-ix   { color: #dcdcdc; font-weight: 600;
                        padding: 1px 6px; border-radius: 3px;
                        background: #242424; border: 1px solid #343434;
                        font-size: 10px; letter-spacing: .02em; }
.h3d-status .h3d-pill { color: #4fff8f; font-weight: 600;
                        padding: 1px 6px; border-radius: 3px;
                        background: #1d3a26; border: 1px solid #2c4a38;
                        font-size: 10px; letter-spacing: .02em; }

.h3d-status.is-rendering { color: #c8ffd6; border-color: #2c4a38; }
.h3d-status.is-rendering::before { background: #4fff8f;
                                   animation: h3d-pulse 1.6s ease-in-out infinite; }
.h3d-status.is-rendering .h3d-c   { color: #c8ffd6; }

.h3d-status.is-done { color: #9ed7b5; border-color: #2c4a38; }
.h3d-status.is-done::before { background: #4fff8f; }
.h3d-status.is-done .h3d-c   { color: #9ed7b5; }

.h3d-status.is-error { color: #ffb8b8; border-color: #4a2c2c; }
.h3d-status.is-error::before { background: #ff8080; }
.h3d-status.is-error .h3d-c   { color: #ffb8b8; }

@keyframes h3d-pulse {
  0%, 100% { opacity: 1;    box-shadow: 0 0 0   #4fff8f00; }
  50%      { opacity: .4;   box-shadow: 0 0 6px #4fff8f88; }
}
.h3d-empty { font-size: 10.5px; color: #6f6f6f; padding: 4px 0; }

/* ========= 功能规划：左栏提示词 + 右栏分镜预览 =========
 * ★ 不是独立版块：整块挂在「8 分镜时间线」的 .h3d-bd 里，跟着时间线一起
 *   折叠/展开，所以没有 .h3d-sec、没有序号徽标、没有「功能规划」标题。
 *   只用一条虚线 + 一行小标题跟上面的时间线轨道分开，视觉上属于它的下一层。
 * 两栏等宽 flex，gap 9px（跟面板其它处一致）。每栏都是带卡片头的容器：
 *   - 头：shot_id 中性徽标 + caption + 右侧小状态字
 *   - 体：左=滚动文本 / 右=video 元素或首帧 placeholder
 *
 * ---- 第 18 轮：Liquid Glass ----
 * ★ 关键判断：**两栏一视同仁会出事**。
 *   左栏内容区是可编辑 textarea、右栏是 video 控件 —— 都有原生交互。
 *   整卡铺遮罩会挡住点击/输入/拖选择，那就不是美化是破坏。
 *   所以按 Apple "玻璃是功能层"的原则分开处理：
 *     · 左栏（提示词）→ 玻璃**只做外框**：圆角 + 边缘高光 + focus 时点亮
 *       边框。不加遮罩（会挡输入）。
 *     · 右栏（视频/首帧预览）→ 这是**只读媒体**，可以整卡遮罩：
 *       hover 时浮起一层 Liquid Glass，居中显示本段信息（与参考图卡片同语言）。
 *   左栏 textarea / 右栏 video 的交互区一律不吃 pointer-events。 */
.h3d-plan-inline { margin-top: 9px; padding-top: 8px;
                   border-top: 1px dashed #2e2e2e; min-width: 0; }
/* ★ 2026-09-19：标题、语言/写回工具栏、重选（或 parser 段按钮）**同处一行**。
   此前工具栏（.h3d-plan-tools）自己占第二行，导致「重选」孤挂在标题行右端、
   与下方 4 个胶囊错开一行，视觉上不成组（用户截图反馈「按钮没水平对齐」）。
   现在合并成一条水平线；flex-wrap 兜底：节点拖窄时按行折行，仍是左对齐。
   gap 用「行 6px / 列 8px」两值写法，折行后行距不会撑成 8px。 */
.h3d-plan-bar { display: flex; align-items: center; gap: 6px 8px;
                flex-wrap: wrap; margin-bottom: 8px; min-width: 0; }
.h3d-plan-bar-lb { font-size: 10px; color: #7d7d7d; letter-spacing: .02em;
                   margin-right: 2px; }
.h3d-plan-bar .h3d-btn { margin-left: auto; }
.h3d-plan-grid { display: flex; gap: 9px; min-width: 0; }
.h3d-plan-card { flex: 1 1 0; min-width: 0; display: flex; flex-direction: column;
                 border: 1px solid rgba(255,255,255,.10);
                 border-radius: var(--h3d-r-lg); background: rgba(255,255,255,.03);
                 overflow: hidden;
                 box-shadow: var(--h3d-rim);
                 transition: border-color var(--h3d-dur-fast),
                             box-shadow var(--h3d-dur); }
/* 选中态：accent 绿描边 + 外发光。第 18 轮从 4px 圆角升到 12px，
 * 圆角变大后描边也要跟着柔化，硬描边在大圆角上会显脏。 */
.h3d-plan-card.is-selected { border-color: rgba(79,255,143,.45);
                             box-shadow: var(--h3d-rim),
                                         0 0 0 1px rgba(79,255,143,.22) inset,
                                         0 0 10px rgba(79,255,143,.12); }
.h3d-plan-card .h3d-plan-hd { display: flex; align-items: center; gap: 7px;
                              padding: 5px 9px;
                              background: rgba(255,255,255,.03);
                              border-bottom: 1px solid rgba(255,255,255,.07);
                              font-size: 10.5px; color: #9a9a9a; }
.h3d-plan-card .h3d-plan-hd .lb { font-size: 10px; color: #7d7d7d; }
.h3d-plan-card .h3d-plan-hd .who { color: #dcdcdc; font-weight: 600;
                                   font-variant-numeric: tabular-nums; }
/* 段号徽标：玻璃压印胶囊（第 18 轮从 3px 方块改成胶囊） */
.h3d-plan-card .h3d-plan-hd .who .h3d-ix { color: #e8e8e8; font-weight: 600;
                                            padding: 1px 7px;
                                            border-radius: var(--h3d-r-pill);
                                            background: var(--h3d-press-bg);
                                            border: 1px solid var(--h3d-press-bd);
                                            box-shadow: var(--h3d-rim-hi);
                                            font-size: 10px; letter-spacing: .02em; }
/* 分镜时间统计（取自 PACK 段头，如 11s / 11s+1.6=12.6），段号徽标右侧 */
.h3d-plan-card .h3d-plan-hd .who .h3d-time { color: #8fe0b0; font-weight: 600;
                                             font-size: 10px;
                                             font-variant-numeric: tabular-nums; }
.h3d-plan-card .h3d-plan-hd .stat { margin-left: auto; font-size: 9.5px;
                                     color: #6f6f6f; letter-spacing: .02em; }
.h3d-plan-card .h3d-plan-hd .stat.is-ok  { color: #4fff8f; }
.h3d-plan-card .h3d-plan-hd .stat.is-bad { color: #ff8080; }
.h3d-plan-card .h3d-plan-bd { padding: 8px 10px 9px; flex: 1 1 auto;
                              min-height: 0; min-width: 0; }
.h3d-plan-card .h3d-plan-bd.is-prompt {
  font-size: 10.5px; line-height: 1.6; color: #c8c8c8;
  max-height: 230px; overflow-y: auto; white-space: pre-wrap;
  word-wrap: break-word; font-variant-numeric: tabular-nums;
  scrollbar-width: thin; scrollbar-color: #343434 transparent;
}
.h3d-plan-card .h3d-plan-bd.is-prompt::-webkit-scrollbar { width: 6px; }
.h3d-plan-card .h3d-plan-bd.is-prompt::-webkit-scrollbar-thumb {
  background: #343434; border-radius: 3px; }
.h3d-plan-card .h3d-plan-bd.is-empty {
  display: flex; align-items: center; justify-content: center;
  min-height: 96px; color: #6f6f6f; font-size: 10.5px; text-align: center;
}
/* 预览区：内容（video / 首帧图 / 占位）**撑满整块**，与左栏提示词框等高。
   左栏 textarea 撑多高，右栏卡片就被 flex 拉到同高；预览区用 flex:1 吃掉卡片
   除标题外的全部高度，video/img 用 width/height:100% + object-fit:contain
   保持比例铺满（★ max-height:100% 在 flex item 上不可靠，实测视频会卡在旧高度）。 */
.h3d-plan-card .h3d-plan-bd.is-preview { padding: 0; position: relative;
                                          background: #0c0c0c;
                                          min-height: 158px;
                                          flex: 1 1 auto;
                                          overflow: hidden; }
.h3d-plan-card .h3d-plan-bd.is-preview video,
.h3d-plan-card .h3d-plan-bd.is-preview img {
  display: block; width: 100%; height: 100%;
  object-fit: contain; background: #000;
}

/* ---- ★ 预览卡片的整卡遮罩（第 18 轮：与参考图卡片同一套交互语言）----
 * 这是只读媒体，可以放心铺遮罩。结构和 .h3d-ref .veil 完全同构：
 *   .pveil 整卡玻璃浮层 → .ptop 顶部细行 / .pkind 中间大字 / .pacts 底部操作
 * 复用同一组设计令牌，所以两处玻璃**永远一致**（改令牌即同时生效）。
 * ★ 用"卡片的 :hover"驱动 .pveil 的透明度，而不是给 video 套 hover，
 *   因为遮罩和 video 是同一个 .is-preview 里的兄弟/父子，hover 卡片更稳。 */
.h3d-plan-card .h3d-plan-bd.is-preview .pveil {
  position: absolute; inset: 0; z-index: 3;
  display: flex; flex-direction: column;
  padding: 9px 11px;
  /* ★ 第 42 轮（2026-09-17）：这层铺在首帧参考图 / 视频画面上，原来的模糊
     滤镜会把画面糊掉（第 23 轮只处理了「已渲染」分支，未渲染分支漏了）。
     同样改成只压暗、不模糊 —— 文字/按钮靠 text-shadow + 白描边保持可读。
     ★ 本块内不得再出现任何模糊滤镜 —— 见回归断言。 */
  background: rgba(0, 0, 0, .26);
  box-shadow: var(--h3d-rim-lg);
  opacity: 0; pointer-events: none;
  transition: opacity var(--h3d-dur) var(--h3d-ease); }
.h3d-plan-card .h3d-plan-bd.is-preview:hover .pveil { opacity: 1; }
/* ★ 只有按钮复活 pointer-events —— 遮罩本体放行，video 的原生 controls
   （播放/进度条/音量）在 hover 时仍可点。这条不加，遮罩一浮上来就成"死玻璃"。 */
.h3d-plan-card .pveil .pact { pointer-events: auto; }
/* 顶部细行：左=渲染状态，右=段号 */
.h3d-plan-card .pveil .ptop { display: flex; align-items: center;
                              justify-content: space-between;
                              font-size: 10px; line-height: 1.5;
                              color: rgba(255,255,255,.62);
                              font-variant-numeric: tabular-nums;
                              text-shadow: var(--h3d-txt-shadow); }
.h3d-plan-card .pveil .ptop .pst { color: rgba(255,255,255,.92); font-weight: 600; }
.h3d-plan-card .pveil .ptop .pst.is-ok  { color: var(--h3d-acc); }
.h3d-plan-card .pveil .ptop .pst.is-bad { color: var(--h3d-bad); }
.h3d-plan-card .pveil .ptop .psid { color: rgba(255,255,255,.5); }
/* 中间大字：本段操作提示（主信息，最大最亮） */
.h3d-plan-card .pveil .pkind { flex: 1 1 auto;
                               display: flex; flex-direction: column;
                               align-items: center; justify-content: center;
                               gap: 4px;
                               font-size: 13px; font-weight: 700; line-height: 1.3;
                               color: #fff; text-align: center;
                               letter-spacing: .3px;
                               text-shadow: 0 1px 2px rgba(0,0,0,.55),
                                            0 0 8px rgba(0,0,0,.35); }
.h3d-plan-card .pveil .pkind .sub { font-size: 10px; font-weight: 400;
                                    color: rgba(255,255,255,.6);
                                    letter-spacing: 0;
                                    text-shadow: var(--h3d-txt-shadow); }
/* 底部操作：与参考图卡片同款的"压印"圆钮 */
.h3d-plan-card .pveil .pacts { flex: none;
                               display: flex; align-items: center;
                               justify-content: center; gap: 8px; }
.h3d-plan-card .pveil .pact { min-width: 30px; height: 24px; padding: 0 10px;
                              border-radius: var(--h3d-r-pill);
                              cursor: pointer; flex: none;
                              display: flex; align-items: center;
                              justify-content: center; gap: 4px;
                              background: var(--h3d-press-bg);
                              border: 1px solid var(--h3d-press-bd);
                              box-shadow: var(--h3d-rim-hi);
                              color: #fff; font-size: 10.5px; line-height: 1;
                              text-shadow: var(--h3d-txt-shadow);
                              transition: background var(--h3d-dur-fast),
                                          border-color var(--h3d-dur-fast),
                                          box-shadow var(--h3d-dur-fast),
                                          transform .1s; }
.h3d-plan-card .pveil .pact:hover { background: var(--h3d-press-bg-hi);
                                    border-color: var(--h3d-press-bd-hi);
                                    box-shadow: inset 0 1px 0 rgba(255,255,255,.45),
                                                var(--h3d-lift-sm);
                                    transform: translateY(-1px); }
.h3d-plan-card .pveil .pact:active { transform: translateY(0) scale(.95);
                                     box-shadow: inset 0 1px 2px rgba(0,0,0,.3); }
.h3d-plan-card .pveil .pact.is-play:hover { background: rgba(47,158,68,.72);
                                            border-color: rgba(106,255,160,.8); }
.h3d-plan-card .h3d-plan-bd.is-preview .ph {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 6px; color: #6f6f6f; font-size: 10px;
}
.h3d-plan-card .h3d-plan-bd.is-preview .ph .lb { color: #9a9a9a; font-size: 10.5px; }
.h3d-plan-card .h3d-plan-bd.is-preview .ph img {
  max-height: 55%; max-width: 88%; opacity: .55; object-fit: contain;
  border-radius: 3px; border: 1px solid #2a2a2a;
}
.h3d-plan-card .h3d-plan-bd.is-preview .ph .ic {
  width: 28px; height: 28px; opacity: .35;
}
.h3d-plan-card .h3d-plan-bd.is-preview .ph-row {
  display: flex; align-items: center; gap: 6px;
}

/* 工具栏组（语言 EN/ZH + 写回模式）：**内联**在标题行里，不再自占一行。
   仍是「语言组 ↔ 模式组」14px 的分组间距，只是整组跟着标题走。
   注意别放回左栏卡片内 —— 那样左栏内容区会被推低、右栏视频却紧贴标题。 */
.h3d-plan-tools { display: inline-flex; align-items: center; gap: 14px;
                  flex-wrap: wrap; min-width: 0; }
.h3d-lang-tabs { display: flex; gap: 4px; }
/* 语言标签：与 .h3d-btn 同族的"压印"控件 —— 半透明白填充 + 顶部亮线，
   不各带 backdrop-filter（Apple：不要 glass-on-glass 嵌套）。
   选中态用 accent 绿描边点亮，与参考图卡片的 is-cycle 同一语言。 */
.h3d-lang-tab { flex: 0 0 auto; font: 10px/1 system-ui, sans-serif;
  /* ★ 2026-09-19：与 .h3d-btn 统一 24px 高度 —— 二者现在同排，一高一矮会明显错位。
     原先靠 padding:4px 10px 撑出约 20px，比 .h3d-btn 矮 4px。改用固定高 +
     inline-flex 居中，文字基线不再受 padding 与字体行高互相牵扯。
     左右内边距保持 10px（比 .h3d-btn 的 9px 略宽，胶囊形按钮留白更舒服）。 */
  display: inline-flex; align-items: center; justify-content: center;
  height: 24px; padding: 0 10px;
  color: rgba(255,255,255,.62);
  background: var(--h3d-press-bg);
  border: 1px solid var(--h3d-press-bd);
  border-radius: var(--h3d-r-pill);
  cursor: pointer; letter-spacing: .02em;
  box-shadow: var(--h3d-rim-hi);
  transition: background var(--h3d-dur-fast), border-color var(--h3d-dur-fast),
              color var(--h3d-dur-fast), transform .1s; }
.h3d-lang-tab:hover { color: #fff; background: var(--h3d-press-bg-hi);
                      border-color: var(--h3d-press-bd-hi);
                      transform: translateY(-1px); }
.h3d-lang-tab:active { transform: translateY(0) scale(.95); }
.h3d-lang-tab.is-active { color: #dfffee; background: rgba(47,158,68,.42);
  border-color: rgba(106,255,160,.62);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.28), 0 0 10px rgba(79,255,143,.22); }
/* 左栏内容区 padding 归零：textarea 铺满内容区，边界与右栏视频一致（齐平）。
   textarea 自己的 padding 7px 9px 保证文字不贴边，视觉上不受影响。
   ★ flex column + textarea flex:1 —— 让 textarea **填满**整个内容区：
     内容区（.h3d-plan-bd）在卡片里是 flex:1 1 auto，会被卡片（进而被右栏
     等高逻辑）拉高；textarea 若只是普通 block + min-height:260，就停在
     260~380px，卡片底部剩下一截**没有内容却仍带节点紫底**的空白 —— 用户
     报的「外下边框没跟随文本框」就是这个。填满后 textarea 下边框 = 内容区
     下边框 = 卡片下边框，紫底紧贴文本框，中间不再空一截。 */
.h3d-plan-card .h3d-plan-bd.h3d-ed { padding: 0;
  display: flex; flex-direction: column;
  scrollbar-width: thin; scrollbar-color: #343434 transparent; }
.h3d-plan-card .h3d-plan-bd.h3d-ed::-webkit-scrollbar { width: 6px; }
.h3d-plan-card .h3d-plan-bd.h3d-ed::-webkit-scrollbar-thumb { background: #343434;
  border-radius: 3px; }
/* 整段六段式提示词：一个 textarea 装全文，可纵向拉伸。
   ★ flex:1 1 auto（承接上面 .h3d-ed 的 flex column）—— 高度跟着内容区走，
     而不是钉在自己的 min-height 上。max-height 一并去掉：留着它会在内容区
     更高时把 textarea 卡在 460px，下方又空出一截（同一个病）。 */
.h3d-block-ta { display: block; width: 100%; box-sizing: border-box;
  flex: 1 1 auto; resize: vertical; min-height: 260px;
  background: #151515; color: #cfcfcf;
  border: 1px solid #2e2e2e; border-radius: 3px; padding: 7px 9px;
  font: 11px/1.6 "SF Mono", "Cascadia Code", Consolas, monospace;
  white-space: pre-wrap; word-wrap: break-word; outline: none;
  scrollbar-width: thin; scrollbar-color: #343434 transparent; }
.h3d-block-ta::-webkit-scrollbar { width: 7px; }
.h3d-block-ta::-webkit-scrollbar-thumb { background: #343434; border-radius: 3px; }
.h3d-block-ta:focus { border-color: #3a5a44; background: #181818; }
.h3d-block-ta.is-zh { color: #cdbfae; }

/* 分镜时间线缩略图被选中态：边框 + 内阴影 + 微小放大 */
.h3d-seg.is-plan-selected {
  outline: 1px solid #4fff8f;
  outline-offset: -1px;
  box-shadow: 0 0 0 1px #2c4a3888 inset, 0 0 8px #4fff8f33;
}

/* ========= 版块头：序号徽标 + 标题（顺序即操作顺序）=========
 * 第 18 轮：序号徽标也是"控件"（可点击折叠），换成玻璃压印材质。
 * 15px 的小方块用 6px 圆角（--h3d-r-xs）—— 圆角再大在这么小的尺寸上
 * 会显得像药丸，反而失去"方块数字"的辨识度。 */
.h3d-hd .h3d-ix { flex: none; width: 15px; height: 15px; border-radius: var(--h3d-r-xs);
                  background: var(--h3d-press-bg); border: 1px solid var(--h3d-press-bd);
                  color: #a5a5a5; font-size: 9.5px; font-weight: 700;
                  line-height: 13px; text-align: center; cursor: pointer;
                  user-select: none;
                  box-shadow: var(--h3d-rim-hi);
                  transition: background var(--h3d-dur-fast), color var(--h3d-dur-fast),
                              border-color var(--h3d-dur-fast), transform .1s; }
.h3d-hd .h3d-ix:hover { background: rgba(47,158,68,.82); border-color: var(--h3d-acc);
                        color: #fff; transform: translateY(-1px); }
.h3d-hd .h3d-ix:active { transform: translateY(0) scale(.94); }

/* ========= 4) 输出规格：一行一设置，标签 / 控件 / 结果（同样不套卡片） ========= */
.h3d-fields { display: flex; flex-direction: column; gap: 6px;
              padding: 1px 0 2px; background: transparent;
              border: 0; border-radius: 0; min-width: 0; }
.h3d-field { display: flex; align-items: center; gap: 7px; min-width: 0;
             flex-wrap: wrap; }
.h3d-field > .lb { flex: none; width: 52px; font-size: 10px; color: #7d7d7d; }
.h3d-field .unit { font-size: 10px; color: #6f6f6f; white-space: nowrap; }
.h3d-field .arrow { font-size: 11px; color: #4a4a4a; padding: 0 1px; }
/* 多行字段：标签在上、控件占满整行（textarea 塞进 52px 标签右边太窄） */
.h3d-field.is-block { flex-direction: column; align-items: stretch; gap: 3px; }
.h3d-field.is-block > .lb { width: auto; }
/* 输入框：玻璃语言里的"内凹"面 —— 与按钮的"压印外凸"互补。
   inset 阴影代替实心 border，聚焦时 accent 绿描边 + 外发光点亮
   （与卡片 is-selected / 标签 is-active 同一套 emphasis 语言）。 */
.h3d-in { height: 22px; padding: 0 7px; font-size: 10.5px; color: #eaeaea;
          background: rgba(0,0,0,.32);
          border: 1px solid rgba(255,255,255,.10);
          border-radius: var(--h3d-r-xs);
          outline: none; font-family: inherit;
          font-variant-numeric: tabular-nums;
          box-shadow: inset 0 1px 2px rgba(0,0,0,.42);
          transition: border-color var(--h3d-dur-fast),
                      box-shadow var(--h3d-dur-fast),
                      background var(--h3d-dur-fast); }
.h3d-in:hover { border-color: rgba(255,255,255,.20); }
.h3d-in:focus { background: rgba(0,0,0,.46);
                border-color: rgba(106,255,160,.66);
                box-shadow: inset 0 1px 2px rgba(0,0,0,.42),
                            0 0 0 2px rgba(79,255,143,.16); }
.h3d-in:disabled { opacity: .4; cursor: default; }
.h3d-in.is-num { width: 60px; text-align: right; }
select.h3d-in.is-sel { width: 150px; cursor: pointer; }
/* 多行文本 / 勾选：和单行控件共用同一套视觉，避免节点上"两种风格" */
textarea.h3d-in { height: auto; min-height: 42px; padding: 4px 7px;
                  resize: vertical; line-height: 1.45; width: 100%;
                  font-family: inherit; }
.h3d-chk { display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
           font-size: 10.5px; color: #c8c8c8; user-select: none; }
.h3d-chk input { width: 13px; height: 13px; margin: 0; cursor: pointer;
                 accent-color: #2f9e44; }
.h3d-chk:hover { color: #eee; }
.h3d-out { font-size: 11.5px; font-weight: 600; color: #4fff8f;
           font-variant-numeric: tabular-nums; letter-spacing: .01em; }
.h3d-out.is-mute { color: #5f5f5f; font-weight: 400; }
.h3d-note { font-size: 9.5px; color: #666; line-height: 1.5;
            padding-left: 75px; }
.h3d-note.is-warn { color: #c9a227; }
/* ★ 第 30 轮：致命诊断（整段 PACK 正文为空）比普通提醒更扎眼 —— 红字。
   以前这类坏数据完全不显示，用户只能对着空提示词框猜。 */
.h3d-note.is-warn.is-fatal { color: #ff8080; font-weight: 600; }
.h3d-note.is-link { cursor: pointer; }
.h3d-note.is-link:hover { color: #4fff8f; }
`;

function injectCss() {
  if (document.getElementById("h3d-css")) return;
  const style = document.createElement("style");
  style.id = "h3d-css";
  style.textContent = CSS;
  document.head.appendChild(style);
}

/* ------------------------------------------------------------------ */
/* 小工具                                                              */
/* ------------------------------------------------------------------ */
function h(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

/** 渲染分镜时间线状态行内容。
 *  结构化文本（"第 N/M 段 · SXX · NNN 帧 · task · 参考图 N 张"）拆成 caption + shot_id 徽标 + 绿色 task 徽标；
 *  纯文字（"空闲" / "↻ 断点渲染：..." / 错误信息）走兜底，保持单文本节点。
 *  同时负责加 is-rendering / is-done / is-error 类名。 */
function _renderStatusText(status, text, running) {
  status.replaceChildren();
  status.className = "h3d-status"
    + (running ? " is-rendering"
       : /^✓/.test(text) ? " is-done"
       : /^⚠/.test(text) ? " is-error" : "");

  let body = String(text || "");
  let prefix = "";
  const m = body.match(/^([✓⚠↻])\s*/);
  if (m) { prefix = m[1]; body = body.slice(m[0].length); }

  if (prefix) status.appendChild(document.createTextNode(prefix + " "));

  const shotRe = /^S\d+$/;
  const taskRe = /^(r2v|t2v|i2v|t2i|r2v_edit|t2v_edit)$/;
  if (body.includes(" · ")) {
    const parts = body.split(/\s*·\s*/).filter(Boolean);
    for (const p of parts) {
      if (shotRe.test(p)) {
        status.appendChild(h("span", "h3d-ix", p));
      } else if (taskRe.test(p)) {
        status.appendChild(h("span", "h3d-pill", p));
      } else {
        status.appendChild(h("span", "h3d-c", p));
      }
    }
  } else if (body) {
    status.appendChild(document.createTextNode(body));
  }
}

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name) || null;
}

function graphNodes() {
  const g = app.graph;
  return (g && (g._nodes || g.nodes)) || [];
}

function upstream(node, inputName) {
  const input = (node.inputs || []).find((i) => i.name === inputName);
  if (!input || input.link == null) return null;
  const g = app.graph;
  let link = null;
  try {
    if (g.links && typeof g.links.get === "function") link = g.links.get(input.link);
    else if (g.links) link = g.links[input.link];
  } catch (e) {
    link = null;
  }
  if (!link) return null;
  const originId = link.origin_id != null ? link.origin_id
    : (link.originId != null ? link.originId : link[1]);
  try {
    return g.getNodeById(originId);
  } catch (e) {
    return null;
  }
}

/** ref_image_0..8 输入口 → 上游 LoadImage 选的文件名 */
function refSlotImages(node) {
  const out = [];
  for (let i = 0; i < 9; i++) {
    const input = (node.inputs || []).find((x) => x.name === `ref_images.ref_image_${i}`);
    if (!input) continue;
    const src = upstream(node, input.name);
    let file = "";
    if (src) {
      const w = src.widgets && src.widgets[0];
      if (w && typeof w.value === "string") file = w.value;
    }
    out.push({ slot: i + 1, file, linked: !!input.link, node: src });
  }
  return out;
}

/** 一段的「一句话摘要」—— 时间线缩略图下方的 cap、右键菜单的复制项都用它。
 *
 *  ★ 数据源必须是 /h3/pack_preview **真实返回**的键，不能想当然。
 *    后端 _pack_preview 下发的是：
 *      - ``full_fields``  : {subject_definitions, summary, retention_analysis,
 *                           detailed_description, overall_soundscape,
 *                           non_diegetic_music}（FIELD_ORDER）
 *      - ``prompt``       : detailed_description 正文（[Shot N] 脚本形态下是全文）
 *    它**从来没有** ``fields`` 这个键。以前的实现读 seg.fields.scene/action →
 *    永远 undefined → 时间线上"分镜提示词"整排不渲染（用户报的「不见了」）。
 *    这里按「full_fields 的具体字段 → prompt 正文 → 空」逐级回落，两种形态
 *    （六字段 PACK / [Shot N] 脚本 PACK）都能出摘要。
 *    仍然保留对老 ``fields`` 的兼容：万一有别的调用方传进来，不至于静默失效。
 */
function segBrief(seg) {
  if (!seg) return "";
  const norm = (s) => String(s || "").replace(/\s+/g, " ").trim();
  const f = seg.fields || {};
  const ff = seg.full_fields || {};
  const direct = norm(f.scene || f.action || f.subject_definitions);
  if (direct) return stripTaskPrefix(direct);
  // 六字段形态：summary 最像"这一段讲什么"，其次 detailed_description
  // ★ summary 规范要求以 [reference generation] 之类开头（后端校验会查），
  //   但时间线那一行容不下这个前缀 —— 展示层剥掉，数据层不动。
  const fromFields = norm(ff.summary || ff.detailed_description
    || ff.subject_definitions);
  if (fromFields) return stripTaskPrefix(fromFields);
  // [Shot N] 脚本形态：full_fields 只有 detailed_description 被填，
  // 拿不到时退回解析器给的 prompt 正文
  return stripTaskPrefix(norm(seg.prompt));
}

/* 官方任务前缀：后端 summary 规范**要求**以 [xxx] 开头
 * （见 prompt_pack.TASK_PREFIXES / 校验里那条
 *  "summary 没有用官方任务前缀开头"），所以原文必须保留、绝不能改。
 * 但时间线缩略图那一行只有一行高度、宽度又窄，前缀纯属噪音 ——
 * 用户要的是"这段讲什么"，不是"这是什么任务类型"。
 * → 只在**展示层**剥掉，数据层一个字不动。
 * ★ 只认行首 + 只认后端登记过的那几种前缀，不能全局删中括号
 *   （正文里可能有作者自己写的中括号，那是有意义的）。 */
const TASK_PREFIX_RE = /^\s*\[\s*(?:keyframe completion|reference generation|video editing|video continuation|audio reuse|audio reference)[^\]]*\]\s*/i;

/** 剥掉展示用的任务前缀。可能重复出现（"video continuation + reference generation"
 *  是一个前缀；但若有人写了两层，这里循环剥干净）。 */
function stripTaskPrefix(text) {
  let s = String(text || "");
  // 最多剥 3 层，防病态输入把正则跑成死循环
  for (let i = 0; i < 3; i++) {
    const next = s.replace(TASK_PREFIX_RE, "");
    if (next === s) break;
    s = next;
  }
  return s;
}

/** 本段**全部会送进模型的文本**拼一起 —— 用于抠 <Picture N> / <Video K>。
 *
 *  ★ 为什么不能只看 seg.prompt（第 19 轮的真 bug）：
 *    `routes.py` 把 `seg.prompt` 取成 `fields["detailed_description"]`，
 *    而**角色定义块 `subject_definitions` 不在这个字段里** ——
 *    `<Picture 1>/<Picture 2>/<Picture 4>` 这些角色引用全写在那个块。
 *    只扫 seg.prompt 就永远抠不到角色图（实测 manifest：
 *    detailed_description 只含 <Picture 3> 场景，
 *    subject_definitions 才含 1,2,3,4）。
 *    后端 `_picture_slots()` 扫的是全部字段 —— 前端口径必须对齐，
 *    否则「后端确认识别到了、前端显示识别不到」的鬼打墙。 */
function segAllText(seg) {
  if (!seg) return "";
  const parts = [];
  const push = (o) => {
    if (o && typeof o === "object") {
      Object.keys(o).forEach((k) => {
        const v = o[k];
        if (typeof v === "string" && v) parts.push(v);
      });
    }
  };
  push(seg.full_fields);
  push(seg.full_fields_zh);
  push(seg.fields);
  if (typeof seg.prompt === "string" && seg.prompt) parts.push(seg.prompt);
  return parts.join("\n");
}

/** ★★ 本段**逐段变化**的字段拼文本 —— 剔掉 `subject_definitions`。
 *
 *  为什么必须剔（第 19 轮实测，这是"缩略图全都一样"的真根因）：
 *  `subject_definitions`（角色定义块）是**全片共享**的一块文本，每一段里
 *  都原样重复，且里面一口气写着 `<Picture 1> … <Picture 2> … <Picture 3> …
 *  <Picture 4>`。于是：
 *    · 只扫 seg.prompt（= detailed_description）→ 抠不到角色图（旧 bug 1）
 *    · 改成扫「全部字段」（含这个共享块）→ 每段都得到同一组 [1,2,3,4]
 *      → 排序后每段都取同一个槽 → **还是每段同图**（换了个层复发）
 *  两者都不对。**真正能区分的信号只在逐段变化的字段里**：
 *  `detailed_description` / `retention_analysis` / `summary` —— 这些才是
 *  "这一段在哪儿、拍到了谁、用了什么道具"。
 *
 *  所以：缩略图取图优先走「逐段字段」，共享块只作为**最后**的兜底。
 */
function segPerShotText(seg) {
  const parts = [];
  try {
    const put = (o) => {
      if (!o || typeof o !== "object") return;
      Object.keys(o).forEach((k) => {
        if (k === "subject_definitions") return;   // ★ 共享块，剔除
        const v = o[k];
        if (typeof v === "string" && v) parts.push(v);
      });
    };
    put(seg.full_fields);
    put(seg.full_fields_zh);
    put(seg.fields);
  } catch (e) { /* 取不到就当空 */ }
  // prompt = detailed_description 正文（本身就是逐段字段），也纳入
  if (typeof seg.prompt === "string" && seg.prompt) parts.push(seg.prompt);
  return parts.join("\n");
}

/** 本段逐段字段里出现的 <Picture N>（按出现顺序、去重）。
 *  这是「按剧情匹配」的主信号 —— 每段不同才是正常的。 */
function perShotPicSlots(seg) {
  const out = [];
  const tags = segPerShotText(seg).match(/<Picture\s+(\d+)\s*>/gi) || [];
  tags.forEach((t) => {
    const n = Number(String(t).replace(/\D+/g, ""));
    if (n > 0 && out.indexOf(n) < 0) out.push(n);
  });
  return out;
}

/* ---- 「按分段剧情」给缩略图排序所需的两个信号（第 20 轮）----------- *
 *
 * 官方 PACK 的字段分工（本机实测 60616 字符双语 PACK，4 段）：
 *
 *   | 字段                   | 谁在演 / 在哪儿演        | 逐段变化 |
 *   |------------------------|--------------------------|----------|
 *   | subject_definitions    | <Subject K> 定义在 <Picture M> | ✗ 全片共享 |
 *   | summary                | <Subject K> 指人          | ✓ 剧情   |
 *   | detailed_description   | <Subject K> 指人、<Picture M> 指景 | ✓ 剧情 |
 *   | retention_analysis     | 把本段保留的主体**全列一遍** | ✓ 但近似均匀 |
 *
 * ★ 两点必须记住：
 *   ① 剧情字段里指人用的是 **<Subject K>**，不是 <Picture M> ——
 *      所以必须先拿共享块建 Subject→Picture 映射，否则统计不到角色。
 *   ② retention_analysis **不计入**权重：它是合规清单，每段都把
 *      <Subject 1/2/3> 和 <Picture 1/2/4> 列满，区分度趋近于零，
 *      算进去只会把剧情差异抹平。
 */

/** 从共享的 subject_definitions 里抠 <Subject K> → <Picture M> 映射表。
 *  官方写法是**一行一条定义**（中英皆然），形如：
 *    <Subject 2> is the person in <Picture 2> (Zhang San, S2; ...)
 *    <Subject 2> 是 <Picture 2> 中的人（张三，S2；...）
 *  取行内**第一个** <Picture M> 作映射；同一 Subject 只建一次
 *  （后面还有 "<Picture 1> is the appearance reference for <Subject 1>" 这类
 *   反向引用句，不能让它们覆盖正确映射）。抠不出返回空表，不阻断缩略图。 */
function subjectPicMap(seg) {
  const map = new Map();
  try {
    const ff = seg.full_fields || {};
    const fz = seg.full_fields_zh || {};
    const src = String(ff.subject_definitions || fz.subject_definitions || "");
    if (!src) return map;
    src.split(/\n+/).forEach((line) => {
      const sk = line.match(/<\s*Subject\s*(\d+)\s*>/i);
      if (!sk) return;                       // 没有 <Subject K> 的行不是定义行
      const pm = line.match(/<\s*Picture\s*(\d+)\s*>/i);
      if (!pm) return;
      const k = Number(sk[1]);
      if (!map.has(k)) map.set(k, Number(pm[1]));
    });
  } catch (e) { /* 抠不出就当没有映射 */ }
  return map;
}

/** 本段的「剧情权重」：槽号 → 该槽在本段剧情里被提及的次数。
 *
 *  统计范围只有 summary + detailed_description（逐段剧情），
 *  口径 = <Subject K> 出现次数（经映射换算成槽号）+ <Picture M> 出现次数。
 *
 *  为什么这个能区分段落：同一组参考图在不同段里的戏份不同 ——
 *  S01 开场是王总踹门，S03 是张三算账，各自的主角提及次数自然不同。
 *  返回空 Map = 本段剧情里一个引用都没有 → 调用方退回「保原序」。 */
function segPlotPicScores(seg) {
  const scores = new Map();
  const bump = (n) => {
    if (!(n > 0)) return;
    scores.set(n, (scores.get(n) || 0) + 1);
  };
  try {
    const ff = seg.full_fields || {};
    const fz = seg.full_fields_zh || {};
    const f = seg.fields || {};
    const text = [ff.summary, ff.detailed_description,
                  fz.summary, fz.detailed_description,
                  f.summary, f.detailed_description]
      .filter((v) => typeof v === "string" && v).join("\n");
    if (!text) return scores;
    const subjToPic = subjectPicMap(seg);
    (text.match(/<\s*Subject\s*(\d+)\s*>/gi) || []).forEach((t) => {
      bump(subjToPic.get(Number(String(t).replace(/\D+/g, ""))));
    });
    (text.match(/<\s*Picture\s*(\d+)\s*>/gi) || []).forEach((t) => {
      bump(Number(String(t).replace(/\D+/g, "")));
    });
  } catch (e) { /* 抠不出就当没有剧情权重 */ }
  return scores;
}

/** 本段"角色定义"里引用的 <Picture N> 集合 —— 这些是**角色图**。
 *
 *  ★ 为什么要把角色图挑出来：① 没有作者分类元数据时，它是判定"这张图是不是人"
 *    的兜底依据；② 用户口径是「首选角色人物」，所以这个集合直接决定排序优先级。
 *  ⚠ 但**不能**直接拿它当缩略图：subject_definitions 每段一字不差，
 *    直接取它只会让每段同图（第 19 轮踩过）。所以它只参与"是不是人"的判定，
 *    真正的区分靠 orderPicsForThumb 里的逐段引用 + 剧情权重。
 *  来源：subject_definitions（中英任一版）。抠不到返回空集，不影响顺序。 */
function segSubjectPicSlots(seg) {
  const out = new Set();
  try {
    const ff = seg.full_fields || {};
    const fz = seg.full_fields_zh || {};
    const src = String(ff.subject_definitions || fz.subject_definitions || "");
    if (!src) return out;
    // 按句切分，只看含 <Subject K> 的那些句子 —— 角色定义块是一条一条列出来的
    const sentences = src.split(/(?<=\.)\s+/);
    for (const s of sentences) {
      if (!/<Subject\s+\d+>/i.test(s)) continue;
      (s.match(/<Picture\s+(\d+)\s*>/gi) || []).forEach((t) => {
        const n = Number(String(t).replace(/\D+/g, ""));
        if (n > 0) out.add(n);
      });
    }
  } catch (e) { /* 抠不出就当没有，不阻断缩略图 */ }
  return out;
}

/* ---- 参考图槽位的「人 / 景」判定 ---------------------------------- *
 * 缩略图要按剧情匹配，就必须知道哪个槽是**人**（角色）、哪个是**景/物**。
 * 两个信息源，优先级从高到低：
 *
 *   ① 作者手填的分类（`ref_classify` → {slot, kind, name}）—— 最权威
 *   ② 名字里的语义线索 —— 因为①**不可靠**：实测真实工作流里
 *      slot 4 = "路人甲（古装男·krea2_identity_edit）" 却被标成"场景"，
 *      slot 5 是空占位（kind="" name=""）。只看 kind 会把一个"人"
 *      当成环境图排到最前面 —— 正好违背用户"匹配角色/环境/道具"的意图。
 *
 * 所以判定规则是「**先看名字，再看 kind**」：名字里有人物词 → 算人（角色），
 * 无论作者把它标成什么；名字里没有线索时才信 kind。 */

/** 名字里的人物线索。中英都认：中文靠词，英文靠整词（\\b 防止 man 命中 manner）。 */
const _PERSON_NAME_RE = new RegExp([
  "角色", "人物", "男主", "女主", "路人", "群众", "演员", "主持", "老[爷师板婆]",
  "小王", "小姐", "先生", "女[士人]", "少女", "家伙",
  "\\b(?:man|woman|boy|girl|guy|lady|male|female|person|people|character|"
  + "actor|actress|subject|portrait|headshot)\\b",
].join("|"), "i");

/** 名字里的环境/道具线索：命中这些就明确**不是**人（避免"古代卧房"被判成人）。 */
const _PLACE_NAME_RE = new RegExp([
  "场景", "环境", "背景", "布景", "室内", "室外", "房间", "卧房", "卧室", "客厅",
  "街道", "街景", "广场", "山", "河", "海", "树", "天空", "建筑", "城", "村",
  "道具", "物件", "武器", "车", "剑", "刀", "灯", "桌子", "椅子", "场景图",
  "\\b(?:scene|scenery|landscape|background|environment|room|interior|exterior|"
  + "street|building|set\\b|prop|object|vehicle|weapon|furniture)\\b",
].join("|"), "i");

/** 这个参考图槽位代表"人"吗？
 *  meta = 作者填的 {slot, kind, name}（可能缺）。 */
function slotRepresentsPerson(meta) {
  const kind = String((meta && meta.kind) || "");
  const name = String((meta && meta.name) || "");
  // ① 名字优先：环境词命中 → 不是人（"古代卧房·夜" 不能被 kind 拽成人）
  if (name && _PLACE_NAME_RE.test(name)) return false;
  // ② 名字里有人物词 → 是人，不管 kind 写什么（修"路人甲被标成场景"）
  if (name && _PERSON_NAME_RE.test(name)) return true;
  // ③ 名字没线索 → 才信作者填的 kind
  if (kind.indexOf("角色") >= 0) return true;
  if (kind.indexOf("场景") >= 0 || kind.indexOf("道具") >= 0) return false;
  return false;   // 未知（含空占位）默认当"非人"，让景/物优先
}

/** 把候选槽位重排成"更适合当缩略图"的顺序。
 *
 *  ★★ 排序总目标（第 20 轮按用户口径修订）：
 *     1. **逐段变化的引用优先** —— 只有它们才能让各段缩略图不同；
 *        共享的角色定义块引用（每段都一样）必须排到最后，
 *        否则无论怎么排，每段首图都会是同一张。（第 19 轮结论，保留）
 *     2. 同一档内 **首选角色人物，其次是环境/场景图**（用户原话）。
 *     3. 同为角色（或同为场景）时，**按本段剧情权重降序** ——
 *        「这段的戏是谁的」由 summary / detailed_description 里
 *        <Subject K> 的提及次数决定（见 segPlotPicScores）。
 *        权重相同则保持原序 → 结果稳定可预期，不引入随机性。
 *
 *  ⚠ 第 19 轮曾按「环境/道具优先、角色最后」排，本机真实 PACK 实测的后果是
 *    **四段首图全是同一个卧房场景**：四段的 detailed_description 都写着
 *    <Picture 3>（卧房），景一优先就全段同图。用户明确要求改成角色优先，
 *    并且要「根据分段剧情分配」，所以这里把主次反转 + 引入剧情权重。
 *
 *  @param seg   本段
 *  @param nums  候选槽位号
 *  @param metas 可选：[{slot, kind, name}]，作者填的分类元数据
 *  @param perShotNums 可选：**逐段字段**里出现的槽位（第 1 档）。
 *         不传则内部用 perShotPicSlots(seg) 现算。
 *         传空数组 = 本段逐段字段一个引用都没有 → 全部落到第 2 档。 */
function orderPicsForThumb(seg, nums, metas, perShotNums) {
  const arr = [];
  (nums || []).forEach((n) => {
    const v = Number(n);
    if (v > 0 && arr.indexOf(v) < 0) arr.push(v);
  });
  if (arr.length <= 1) return arr;

  const metaOf = new Map();
  (metas || []).forEach((m) => {
    if (m && m.slot != null) metaOf.set(Number(m.slot), m);
  });
  const subjPics = segSubjectPicSlots(seg);
  const isPerson = (n) => {
    if (metaOf.has(n)) return slotRepresentsPerson(metaOf.get(n));
    return subjPics.has(n);   // 没有元数据时退回"角色定义块引用过 = 角色图"
  };
  // ★ 本段剧情权重（每段不同）—— 角色之间靠它分出「这段是谁的戏」
  const scores = segPlotPicScores(seg);

  // 第 1 档：本段**逐段字段**里真正引用到的槽（每段不同 → 天然产生差异）
  const perShot = (perShotNums && perShotNums.length)
    ? perShotNums.map(Number).filter((n) => arr.indexOf(n) >= 0)
    : perShotPicSlots(seg).filter((n) => arr.indexOf(n) >= 0);
  const inPerShot = new Set(perShot);

  const tierA = arr.filter((n) => inPerShot.has(n));   // 逐段引用到的
  const tierB = arr.filter((n) => !inPerShot.has(n));  // 只剩共享块引用的

  // 每档内部：① 角色优先（人=0 / 景=1） ② 剧情权重降序 ③ 原序兜底。
  // 显式带上原序下标，就不依赖 sort 的稳定性（老引擎 sort 不稳定）。
  const rank = (list) => {
    const at = new Map();
    list.forEach((n, i) => at.set(n, i));
    return list.slice().sort((a, b) => {
      const ka = [isPerson(a) ? 0 : 1, -(scores.get(a) || 0), at.get(a)];
      const kb = [isPerson(b) ? 0 : 1, -(scores.get(b) || 0), at.get(b)];
      for (let i = 0; i < 3; i++) { if (ka[i] !== kb[i]) return ka[i] - kb[i]; }
      return 0;
    });
  };
  return rank(tierA).concat(rank(tierB));
}

/* ★ 必须与后端 prompt_pack.py 的 _PIC_RE 同语义：
 *    后端 re.compile(r"<\s*Picture\s*(\d+)\s*>", re.I) —— 忽略大小写、尖括号内
 *    允许任意空白。前端若更严格（比如要求 <Picture 后必须有空格、且大小写敏感），
 *    就会出现"胶囊显示一切正常、后端其实根本没解析到这个引用"的鬼打墙。
 *    这里多认 Video/Audio 是前端展示用的（脚本里可能出现），后端目前只解析 Picture。 */
const TOK_RE = /<\s*(Picture|Video|Audio)\s*(\d+)\s*>/gi;

/** 点击提示词胶囊 / 时间线参考引用，让左侧参考图网格里对应槽位高亮并滚入视野。
 *  只做视觉定位，不改任何渲染数据（数据源仍是 PACK 文本）。 */
function highlightRefSlot(node, slot) {
  try {
    const card = document.querySelector('.h3d-ref[data-slot="' + slot + '"]');
    if (!card) return;
    card.classList.add("is-flash");
    try { card.scrollIntoView({ block: "nearest", behavior: "smooth" }); } catch (e) {}
    setTimeout(() => { try { card.classList.remove("is-flash"); } catch (e) {} }, 1100);
  } catch (e) { /* 找不到就算了 */ }
}

/** 构造一颗原子胶囊：<Picture N> / <Video K> / <Audio J>。
 *  slot 编号按本地 PACK 约定 = 标签里的数字（<Picture N> 即 ref_image_{N-1}）。 */
function makeTokenChip(kind, ordinal, slots, node) {
  const el = h("span", "h3d-tok");
  // ★ 只有 <Picture N> 能拿参考图槽位去校验。本包没有 Video/Audio 输入口，
  //   若一并比对，这两种 token 会被永久标记成"缺失"红框 —— 那是误报，不是故障。
  const checkable = kind === "Picture";
  const hit = checkable ? slots.find((s) => s.slot === ordinal && s.file) : null;
  if (hit) {
    const img = document.createElement("img");
    try { img.src = viewUrl(hit.file, "input"); } catch (e) {}
    img.alt = kind + " " + ordinal;
    el.appendChild(img);
  } else if (checkable) {
    el.classList.add("is-missing");   // 该槽位没关联图片：红框提示
  }
  const label = (kind === "Picture" ? "图" : kind === "Video" ? "视频" : "音频") + " " + ordinal;
  el.appendChild(document.createTextNode(label));
  el.title = "<" + kind + " " + ordinal + ">"
    + (hit ? " · " + hit.file
      : checkable ? " · 该参考槽位未关联图片"
      : " · 本包无此类型参考输入口，后端按脚本原样下发");
  el.onclick = (e) => {
    e.stopPropagation();
    if (checkable && node) highlightRefSlot(node, ordinal);
  };
  return el;
}

/** 把提示词里引用到的 <Picture N> / <Video K> / <Audio J> 渲染成胶囊（去重）。
 *
 *  ★ 原先叫 renderTokens，是「文本 + 胶囊」混合渲染，但**从未被任何地方调用**
 *    （死代码）。这里按预览条的实际需要重写：
 *      - 只出胶囊，不重复铺一遍提示词全文 —— 全文就在上面的 textarea 里，
 *        PACK 全文动辄几千字，铺进预览条只会把它挤爆、且淹没了真正要看的信息；
 *      - 同一 token 重复出现只画一次，预览条回答的是"引用了哪些槽位"而非频次；
 *      - 返回缺失槽位清单，交给调用方出警告行。
 *  @returns {{total: number, missing: number[]}} total=去重后引用数，missing=没接图的槽号 */
function renderTokenChips(host, text, node) {
  host.replaceChildren();
  const slots = refSlotImages(node);
  const seen = new Set();
  const missing = [];
  let m;
  TOK_RE.lastIndex = 0;
  while ((m = TOK_RE.exec(text))) {
    const n = Number(m[2]);
    if (!Number.isFinite(n)) continue;
    // ★ TOK_RE 带 i 标志，可能匹配出 picture / PICTURE。必须先规范化成首字母
    //   大写再往下传 —— 否则 makeTokenChip 里的 kind === "Picture" 比较失效，
    //   小写写法的胶囊既不显示缩略图、也不标缺失，整条判定静默失灵。
    const kind = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
    const key = kind + "#" + n;
    if (seen.has(key)) continue;
    seen.add(key);
    host.appendChild(makeTokenChip(kind, n, slots, node));
    // 只有图片槽位才谈得上"接没接图"，Video/Audio 不进缺失清单（见 makeTokenChip）
    if (kind === "Picture" && !slots.some((s) => s.slot === n && s.file)) {
      missing.push(n);
    }
  }
  TOK_RE.lastIndex = 0;
  missing.sort((a, b) => a - b);
  return { total: seen.size, missing };
}

/** 所有提示词胶囊预览条的注册表。
 *
 *  参考图换图 / 断线时，已经写在提示词里的 <Picture N> 状态会变（有图↔没图），
 *  而 textarea 自身不会触发 input 事件 —— 必须有外部刷新入口。
 *  这条对应参照包 minimax_prompt_mentions.js 的 refreshTokenStates()：
 *  那边是在 contenteditable 里就地重刷 chip，这边是 textarea 真源 + 只读预览条，
 *  所以改成注册表统一重绘。
 *  （本包不照搬参照包的 contenteditable 方案：它为此带了 52KB/52 个函数处理
 *   caret 序列化、selection 映射、剪贴板钩子，属于编辑器整体替换，不是定向修。） */
const TOKEN_VIEWS = new Set();

function refreshTokenViews() {
  TOKEN_VIEWS.forEach((v) => {
    // 面板销毁后节点会脱离文档，顺手回收，避免注册表无限增长
    if (!v.host || !v.host.isConnected) { TOKEN_VIEWS.delete(v); return; }
    try { v.render(); } catch (e) { /* 一个预览条坏了别拖垮其余 */ }
  });
}

/* ---- 时间标尺自适应刻度（A2）---- */
function rulerStep(totalSec, px) {
  const pxPerSec = px / Math.max(0.001, totalSec);
  const steps = [0.5, 1, 2, 5, 10, 15, 30, 60];
  for (const s of steps) if (s * pxPerSec >= 46) return s;   // 标签不挤在一起
  return steps[steps.length - 1];
}

function fmtRuler(sec) {
  sec = Math.round(sec * 10) / 10;
  if (sec >= 60) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec - m * 60);
    return m + "m" + (s ? s + "s" : "");
  }
  return sec + "s";
}

/* 一段分镜的**真实时长** = 新增内容 + 锚定尾巴（第 40 轮）。
   ★ 为什么不是只看 new_seconds：H3 连续分镜里，第 N 段生成的视频是
     「上一段末尾 handoff_seconds 的重放 + 本段 new_seconds 的新内容」，
     落盘帧数就是 (new + handoff) × fps —— 段块里的缩略图 / 视频正是这么长。
     标尺要"对应下方分镜缩略图"，就得用同一个口径，否则尺子和缩略图对不齐。
   ★ 字段缺失（老 PACK / 后端没给）时退回 new_seconds，再没有就退回 1s，
     保证 flexGrow 永远是个正数，不会出现 0 宽段块。 */
function segTotalSec(seg) {
  const nw = Number(seg && seg.new_seconds) || 0;
  const ho = Number(seg && seg.handoff_seconds) || 0;
  return nw + ho > 0 ? nw + ho : nw > 0 ? nw : 1;
}

/* ---- 段号选择解析（A5，与后端 _parse_run_segments 同语法："1,3,5-7"）----
   语义：run_segments 是**段号集合**（任选多段，空=全部）。
   后端 `run_from = min(run_set)`：只要勾中最小段号 ≥ N，后段都会跟着重渲
   （首尾锚定耦合，物理事实不可改）。所以这里只维护集合，起点由后端自己算。 */
function parseRunSegments(text, total) {
  const set = new Set();
  if (!text) return set;
  String(text).split(",").forEach((tok) => {
    tok = tok.trim();
    if (!tok) return;
    if (tok.indexOf("-") >= 0) {
      const parts = tok.split("-");
      let a = parseInt(parts[0], 10), b = parseInt(parts[1], 10);
      if (!isNaN(a) && !isNaN(b)) {
        if (a > b) { const t = a; a = b; b = t; }
        for (let i = a; i <= b; i++) if (i >= 1 && i <= total) set.add(i);
      }
    } else {
      const n = parseInt(tok, 10);
      if (!isNaN(n) && n >= 1 && n <= total) set.add(n);
    }
  });
  return set;
}

/* ---- @ 提及菜单（B2，轻量版：不上 contenteditable）---- */
/** 在 prompt_lines 的代理 textarea 上挂 @ 提及菜单。
 *  输入 @ 弹菜单（来自已关联图片的 refSlotImages），↑↓ 选择 / Enter 确认 / Esc 关闭；
 *  确认时把 <Picture N> 插入光标处，并 dispatchEvent("input") 回灌原生 widget
 *  （否则 ComfyUI 原生 widget 的值不会同步，Queue 时拿到旧值）。 */
function attachMentionMenu(ta, node) {
  let menu = null, items = [], cur = 0, atPos = -1;

  function close() {
    if (menu) { try { menu.remove(); } catch (e) {} menu = null; }
    items = []; cur = 0; atPos = -1;
  }

  function position() {
    if (!menu) return;
    const r = ta.getBoundingClientRect();
    const mh = menu.offsetHeight || (Math.min(items.length, 7) * 30 + 8);
    let top = r.top - mh - 6;
    if (top < 4) top = r.bottom + 6;   // 上方放不下就翻到 textarea 下方
    menu.style.top = top + "px";
    menu.style.left = (r.left + 4) + "px";
  }

  function paintCur() {
    if (!menu) return;
    menu.querySelectorAll(".h3d-mention-item").forEach((el, i) =>
      el.classList.toggle("is-cur", i === cur));
    const curEl = menu.children[cur];
    if (curEl && curEl.scrollIntoView) curEl.scrollIntoView({ block: "nearest" });
  }

  function open(at) {
    atPos = at;
    const all = refSlotImages(node);
    // ★ 只列已接图的槽位 —— 这是"引用的槽位一定存在"的第一道保证：
    //   从这里插进去的 <Picture N> 必然有图。空槽绝不进可选列表。
    const slots = all.filter((s) => s.file);
    const q = ta.value.slice(at + 1).replace(/\s+/g, "").toLowerCase();
    items = slots
      .map((s) => ({ slot: s.slot, file: s.file, img: viewUrl(s.file, "input") }))
      .filter((it) => !q || (String(it.slot) === q) || ("p" + it.slot) === q);
    cur = 0;
    menu = h("div", "h3d-mention");
    items.forEach((it, i) => {
      const row = h("div", "h3d-mention-item" + (i === 0 ? " is-cur" : ""));
      const img = document.createElement("img");
      try { img.src = it.img; } catch (e) {}
      row.appendChild(img);
      row.appendChild(document.createTextNode("<Picture " + it.slot + ">"));
      row.appendChild(h("span", "nm", it.file));
      row.onmousedown = (e) => { e.preventDefault(); choose(i); };
      menu.appendChild(row);
    });
    // ★ 空状态：以前无可选项就静默 close()，用户敲了 @ 却毫无反应，
    //   只会以为功能坏了。对齐参照包 minimax_prompt_mentions.js
    //   （mention.emptyNoUpload / mention.emptyFilter）改成明说。
    if (!items.length) {
      const empty = h("div", "h3d-mention-empty");
      empty.textContent = !all.some((s) => s.file)
        ? "还没有槽位接图 —— 先在左侧参考图区选图，再回来输入 @"
        : "没有匹配「" + q + "」的槽位";
      menu.appendChild(empty);
    }
    // 有空槽时告知一声，但**不把它们列成可选项**：列了就能选到一个不存在的
    // <Picture N>，那正好毁掉"引用一定存在"的保证。
    const blanks = all.filter((s) => !s.file).map((s) => s.slot);
    if (blanks.length) {
      menu.appendChild(h("div", "h3d-mention-empty",
        "图片" + blanks.join("、") + " 未接图，未列入可选"));
    }
    document.body.appendChild(menu);
    position();
  }

  function choose(i) {
    const it = items[i];
    if (!it) return close();
    const tag = "<Picture " + it.slot + ">";
    const v = ta.value;
    const afterAt = v.slice(atPos + 1);
    const cut = afterAt.search(/\s/);          // 第一个空白；-1 表示到结尾
    const end = cut < 0 ? afterAt.length : cut;
    ta.value = v.slice(0, atPos) + tag + v.slice(atPos + 1 + end);
    const pos = atPos + tag.length;
    try { ta.setSelectionRange(pos, pos); } catch (e) {}
    ta.dispatchEvent(new Event("input", { bubbles: true }));  // 回灌原生 widget
    close();
  }

  function onInput() {
    paintView();                       // 胶囊预览跟着输入实时刷新
    const pos = ta.selectionStart;
    if (pos == null) return;
    const before = ta.value.slice(0, pos);
    const at = before.lastIndexOf("@");
    if (at < 0 || /\s/.test(before.slice(at + 1))) { if (menu) close(); return; }
    open(at);
  }

  function onKey(e) {
    if (!menu) return;
    // ★ 菜单现在可能为"空状态"（无任何候选项），此时 items.length === 0，
    //   下面的 % items.length 会算出 NaN 让选择彻底失灵 —— 至少要能 Esc 关掉。
    if (!items.length) {
      if (e.key === "Escape" || e.key === "Enter") { e.preventDefault(); close(); }
      return;
    }
    if (e.key === "ArrowDown") { e.preventDefault(); cur = (cur + 1) % items.length; paintCur(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); cur = (cur - 1 + items.length) % items.length; paintCur(); }
    else if (e.key === "Enter") { e.preventDefault(); choose(cur); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  }

  ta.addEventListener("input", onInput);
  ta.addEventListener("keydown", onKey);
  ta.addEventListener("blur", () => setTimeout(close, 120));  // 失焦缓关，允许点击菜单项

  /* ---- 胶囊预览条（B2 补全）：手打的 <Picture N> 到底接没接图，一眼看见 ----
   * @ 菜单只能保证"从这里插进去的一定存在"，挡不住手打一个不存在的编号；
   *  而手打错的后果最隐蔽 —— 后端不报错，H3 自己脑补一个人塞进画面。
   *  这条就是第二道保证：引用了空槽的胶囊显示红框，并在下方明说缺哪几个。
   *  没写任何 <Picture N> 时整条隐藏，不制造噪音。 */
  const view = h("div", "h3d-tokview");
  function paintView() {
    const r = renderTokenChips(view, ta.value || "", node);
    if (!r.total) { view.classList.remove("is-on"); return; }
    view.classList.add("is-on");
    view.appendChild(r.missing.length
      ? h("div", "h3d-tokwarn",
          "⚠ 图片" + r.missing.join("、") + " 还没接图 —— 后端不会报错，"
          + "H3 会自己脑补人物/场景，出来的不是你想要的那个人。")
      : h("div", "h3d-tokok",
          "引用的 " + r.total + " 个槽位都已接图。"));
  }
  TOKEN_VIEWS.add({ host: view, render: paintView });
  paintView();

  return view;   // 交给调用方插到 textarea 下面
}

function viewUrl(name, type, subfolder) {
  let filename = name || "";
  let sub = subfolder || "";
  if (!sub && filename.includes("/")) {
    const cut = filename.lastIndexOf("/");
    sub = filename.slice(0, cut);
    filename = filename.slice(cut + 1);
  }
  let url = `/view?filename=${encodeURIComponent(filename)}&type=${type}`;
  if (sub) url += `&subfolder=${encodeURIComponent(sub)}`;
  return url + `&t=${Date.now()}`;
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

/* ---- 版块骨架：序号徽标 + 标题 + 右侧状态/按钮，点标题折叠 ---- */
function sectionHead(ix, titleText) {
  const hd = h("div", "h3d-hd");
  const badge = h("span", "h3d-ix", String(ix));
  const title = h("span", "h3d-title", titleText);
  const tag = h("span", "h3d-tag", "");
  hd.appendChild(badge);
  hd.appendChild(title);
  hd.appendChild(tag);
  hd.appendChild(h("span", "h3d-spacer"));
  return { hd, badge, title, tag };
}

/** 点标题（或序号徽标）折叠这个版块。折叠后节点高度要跟着缩，所以走 refit。 */
function collapsible(node, bd, ...clickables) {
  const fn = () => {
    bd.classList.toggle("collapsed");
    refit(node);
  };
  clickables.forEach((el) => { if (el) el.onclick = fn; });
}

/* ---- 在整张画布上按节点类型找节点 ---- */
function findNodeByClass(cls) {
  return graphNodes().find((n) => nodeClass(n) === cls) || null;
}

function readWidget(node, name) {
  const w = node ? findWidget(node, name) : null;
  return w || null;
}

function readStr(node, name, dflt) {
  const w = readWidget(node, name);
  const v = w ? w.value : null;
  return (v == null || v === "") ? dflt : String(v);
}

function readNum(node, name, dflt) {
  const v = Number(readWidget(node, name)?.value);
  return Number.isFinite(v) ? v : dflt;
}

/**
 * 写一个 widget 的值。ComfyUI 序列化时直接读 `widget.value`，所以只改它就能
 * 生效；但 DOM 里的 <input>/<select> 不会自己跟着变，得一起刷一遍，
 * 否则用户会看到"我改了数值但框里还是旧的"。
 */
function setWidget(node, name, value) {
  const w = readWidget(node, name);
  if (!w) return false;
  const prev = w.value;
  w.value = value;
  try {
    if (w.inputEl && w.inputEl.value !== undefined) w.inputEl.value = String(value);
    else if (w.element && w.element.value !== undefined) w.element.value = String(value);
  } catch (e) { /* 没有 DOM 元素就算了 */ }
  if (String(prev) !== String(value) && typeof w.callback === "function") {
    try { w.callback(value, app && app.canvas, node, [0, 0], null); } catch (e) { /* 回调可选 */ }
  }
  try { node.setDirtyCanvas(true, true); } catch (e) { /* 老版本没有 */ }
  return true;
}

function isFocused(el) {
  try { return document.activeElement === el; } catch (e) { return false; }
}

/** 内容变了之后把节点高度重新贴合内容：等一帧让 DOM 落位，再按
 *  node.computeSize() 对齐高度（宽度不动，那是用户拖的）。 */
function refit(node) {
  const run = () => {
    try {
      if (typeof node.computeSize !== "function") {
        node.setDirtyCanvas(true, true);
        return;
      }
      const need = node.computeSize();
      const h = Math.max(Math.ceil(need[1] || 0), 80);
      if (Math.abs((node.size[1] || 0) - h) > 1) node.setSize([node.size[0], h]);
      else node.setDirtyCanvas(true, true);
      // 内容高度变了 → 面板 DOM widget 报给布局层的高度也得跟着变，
      // 否则 Nodes 2.0 那边仍按旧高度裁切（时间线只露出一半）。
      try { if (node.__h3dRelayout) node.__h3dRelayout(); } catch (e2) { /* 还没挂上 */ }
    } catch (e) { /* 老版本没有 */ }
  };
  try { requestAnimationFrame(run); } catch (e) { setTimeout(run, 16); }
}

/** 把节点上的原生 widget 收起来（值照旧、只是不画）。
 *
 *  为什么要收：H3Director 自己有十几个原生 widget，ComfyUI 用 canvas 画它们，
 *  样式没法用 CSS 统一 —— 节点上就会同时出现"canvas 画的老控件"和"面板里的
 *  新控件"两套观感。把原生收起、在面板里用同一套 DOM 控件代理，视觉才统一。
 *  只用「布局尺寸归零 + 隐藏 DOM + 标记 hidden」几条路，**不动 w.value / w.type**，
 *  所以存盘、加载、连线都不受影响。
 *
 *  ★ Nodes 2.0（设置项 Comfy.VueNodes.Enabled）走的是另一套布局：
 *    Vue 的 DOMWidgetImpl/BaseWidget 不再问 computeSize，而是问
 *    computeLayoutSize() —— 只把 computeSize 归零的话，Vue 那边照旧给
 *    这个控件留出 minHeight（默认 50px），于是「面板控件 + 原生控件」
 *    两套同时出现，正是「重复项」的来源。
 *    所以这里两条路都堵：legacy 归零 computeSize，Nodes 2.0 归零
 *    computeLayoutSize + computedHeight。反过来关掉 Nodes 2.0 也照样成立，
 *    哪条路存在就走哪条，不存在的那条直接跳过。
 *
 *  ★★ 非 Nodes 2.0 下"堆叠乱象"（十几条极窄的原生控件行手风琴一样叠在
 *    一起）的真身：较新的 ComfyUI 前端（Comfy-Org 维护的 litegraph fork）
 *    把"是否画出来"和"是否占布局位置"拆成了两套独立判断 —— 布局那层
 *    （getLayoutWidgets）只认 widget.hidden 这个显式标志，根本不看
 *    computeSize 是不是归零。以前只归零 computeSize，这些控件在这套新
 *    布局逻辑下依然被当成"可见、要占位"的普通控件，各自留一条自己的
 *    最小原生行高——十几个这种控件挤在一起，就是截图里那种一条条极窄
 *    横线堆成的"手风琴"。补上 w.hidden = true 让新布局逻辑也认账；
 *    同时把 draw 短路成空函数兜底，不管哪个版本的前端，这个控件都不会
 *    再画出任何像素。 */
function hideWidget(node, name) {
  const w = findWidget(node, name);
  if (!w) return null;
  // ★ 不再"只应用一次就跳过"：Nodes 2.0 下 Vue 组件在自己重渲染时会把我们
  //   手动改的行内样式 / computeLayoutSize 覆盖回去（响应式框架重绘时按
  //   自己的状态重新计算 style，不知道也不关心我们手动打的补丁）——只应用
  //   一次的话，一旦被 Vue 覆盖回去就再也不会重新隐藏，节点上原生的"全局
  //   提示词 / 分镜提示词"复活，和面板里的代理控件内容重复。这里的操作全部
  //   是幂等的（重复设成同一个值无副作用），所以允许被反复调用，靠外层的
  //   定时重钉（见 mountPanels 里的 reassertHidden）来对抗 Vue 的覆盖。
  w.__h3dHidden = true;
  // ① legacy litegraph（画布画的 canvas 控件）
  //    返回 [0, 0]（不是 [0, -4]）：隐藏控件本就不占位，-4 会让 node.computeSize
  //    累加一个负值，refit 据此把 node.size[1] 算得比面板 DOM 元素（hhReport）
  //    矮，导致节点紫色背景盖不到底部（时间线/功能规划掉到画布深色上）。
  try { w.computeSize = () => [0, 0]; } catch (e) { /* 只读就算了 */ }
  try { if (w.inputEl) w.inputEl.style.display = "none"; } catch (e) { /* 没有 DOM */ }
  try { if (w.element) w.element.style.display = "none"; } catch (e) { /* 没有 DOM */ }
  // ① .5 新版 litegraph 的布局判断只看这个标志，不看 computeSize；老版本
  //    没这个属性，多写一个也无害（不会被任何地方读到）。
  try { w.hidden = true; } catch (e) { /* 只读就算了 */ }
  // ① .6 双重保险：不管布局层认不认 hidden，画的时候直接短路成不画。
  try { w.draw = () => {}; } catch (e) { /* 只读就算了 */ }
  // ② Nodes 2.0 / Vue 节点：布局走 computeLayoutSize，返回全 0 就不占位
  try {
    if (typeof w.computeLayoutSize === "function" || w.computeLayoutSize === undefined) {
      w.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
    }
    w.computedHeight = 0;
  } catch (e) { /* 只读就算了 */ }
  return w;
}

/** 面板 DOM widget 高度：legacy 走 computeSize、Nodes 2.0 走 computeLayoutSize，
 *  两者都显式给 root 真实高度（否则被压成 50px、时间线被裁）。 */
/* ★ 面板高度硬上限（px）——「加载工作流后整个界面卡死」的总闸门。
 * 背景：2026-09-15 实测，画布上 H3PromptPackParser 的面板内容塌宽时，
 *   root.scrollHeight 按「字符数 × 行高」算出 2,693,023px，节点被撑成
 *   269 万像素高；litegraph 把这个值写回 node.size 并存进工作流 JSON，
 *   于是下次一加载，画布就要为这个 269 万像素高的 DOM widget 做布局与合成，
 *   整界面直接卡住（拖不动、滚轮没反应）。
 * 策略：自然高度超过上限时钳住高度、让面板**内部滚动**，绝不让节点上天。
 *   6000 对现有面板（实测 1300~2500）有充分余量，又远在「会卡」的量级之下。 */
const H3D_PANEL_MAX_H = 6000;

function panelLayout(dom, root, node) {
  // ★ 高度必须量 scrollHeight，不能量 offsetHeight。
  //   legacy（Nodes 2.0 关掉时）litegraph 会把 DOM widget 元素的高**直接
  //   设成自己上一次算出来的尺寸**，于是「量 offsetHeight → 报给布局 → 元素
  //   被设成这个高度 → 再量」形成负反馈：面板被压到 64px，内容 1111px 全被
  //   裁掉，就是"关掉 Nodes 2.0 后节点状态异常"的真身。
  //   scrollHeight 是内容的真实高度，不受容器钳制，两种模式下都成立
  //   （实测：关 1111 / 开 1116，同一份内容）。
  // ComfyUI 给 DOM widget 元素挂了 Tailwind 的 h-full/w-full。
  // h-full = height:100% 会让 scrollHeight 永远 ≥ 容器高度，报尺寸时**只增
  // 不减** —— legacy 下节点会被一路撑到 1831px（Vue 下是 1334）。
  // 内联 height:auto 压过 class，让高度回到"由内容决定"。
  try {
    root.style.height = "auto";
    root.style.minHeight = "0px";
    root.style.maxHeight = "none";
    root.style.boxSizing = "border-box";
  } catch (e) { /* 不支持就算了 */ }
  /* ★ 宽度锚点 —— 折叠引发的"越折越高"就出在这儿：
   *   legacy 的 .dom-widget 容器宽度是**由内容量出来的**（ComfyUI 直接把
   *   style.width 写成量到的值）。平时靠时间线轨道把容器撑到满宽（1080px）；
   *   一旦折叠「8 分镜时间线」，轨道 display:none，容器就塌到剩余最宽内容
   *   的宽度（实测 1080 → 178），参考图 grid 的 auto-fill 从 5 列掉到 1 列，
   *   12 张图竖着堆 → 内容反而从 1357 涨到 2446，节点被越折越高。
   *   按节点宽钉一个 min-width，容器量到的永远是满宽，折叠不再塌。
   *   （Vue 那边容器宽度由布局给定，不会塌，设了也无害。） */
  try {
    const wrap = root.parentElement;
    if (wrap) {
      const nw = Number(node && node.size ? node.size[0] : 0) || 0;
      const want = nw > 240 ? nw - 20 : 0;
      // ★ 原实现里 wrap.__h3dMaxW 是个"只涨不跌"的棘轮：折叠版块导致容器
      //   自然塌宽时靠它兜底（防塌缩到 178px 反而把内容顶高），但代价是
      //   —— 用户把节点从宽拖窄之后，这个历史最大值永远不会跟着变小，
      //   min-width 依旧钉在旧的更宽的值上：节点本身（紫色背景）已经变窄，
      //   面板内容却还留在旧宽度，于是内容右侧露在背景外面 ——「背景不
      //   跟随」在拉窄方向上的真身。
      //   区分两种触发场景：
      //   · 节点宽度本身变了（nw 和上次记的不一样）→ 这是用户在拖边，
      //     必须以最新的 nw 为准，历史棘轮直接归零重记，不能再兜底旧值；
      //   · 节点宽度没变（nw 和上次一样）→ 是内容自己把容器量塌了（比如
      //     折叠了时间线），这时才用历史最宽值兜底，防止 auto-fill 网格
      //     因为容器变窄又把内容顶高。
      const lastNw = wrap.__h3dLastNodeW || 0;
      if (Math.abs(nw - lastNw) > 0.5) {
        wrap.__h3dMaxW = want;
        wrap.__h3dLastNodeW = nw;
      } else {
        const cur = wrap.offsetWidth || 0;
        if (cur > 240 && cur > (wrap.__h3dMaxW || 0)) wrap.__h3dMaxW = cur;
      }
      const target = Math.max(want, wrap.__h3dMaxW || 0, 240);
      if (target > 240) {
        // ComfyUI 每次都会把容器的 style.width 重写成量到的值（折叠后是
        // 178px），但它不动 min-width —— 钉住 min-width 容器才不塌。
        wrap.style.minWidth = target + "px";
        root.style.minWidth = target + "px";
      } else {
        // target 落回 0/240 以下说明连"最小可用宽度"都撑不住了（节点被
        // 拖到极窄），此时不该再留着旧的 minWidth 硬顶着，否则又是背景
        // 比内容窄的老问题——直接清掉交给 CSS 自然值。
        wrap.style.minWidth = "";
        root.style.minWidth = "";
      }
    }
  } catch (e) { /* 拿不到就算了 */ }
  // ★ 只信 scrollHeight，不要 max(scrollHeight, offsetHeight)：
  //   wrapper 上一轮被设成 H 后，父级 flex 会把 root 的 offsetHeight 反向拉到
  //   H，再 max 回去 → 每次 refit +4px 的棘轮（实测 1352 → 1356 → 1376）。
  //   scrollHeight 是内容真实高度，不受容器钳制；只有它拿不到时才退回 offset。
  // ★ 但必须再钳一道 H3D_PANEL_MAX_H：内容塌宽时 scrollHeight 会算出天文数字
  //   （实测 2,693,023），照单全收就把节点和画布一起撑死。超限时给 root 加
  //   max-height + 内部滚动 —— 内容一寸不少，只是改在面板里滚。
  try {
    const natural = Math.max(root.scrollHeight || root.offsetHeight || 0, 80);
    if (natural > H3D_PANEL_MAX_H) {
      root.style.maxHeight = H3D_PANEL_MAX_H + "px";
      root.style.overflowY = "auto";
      root.style.overflowX = "hidden";
    } else {
      root.style.maxHeight = "none";
      root.style.overflowY = "";
      root.style.overflowX = "";
    }
  } catch (e) { /* 拿不到就算了 */ }
  const hh = () => Math.min(
    Math.max(root.scrollHeight || root.offsetHeight || 0, 80), H3D_PANEL_MAX_H);
  // ComfyUI DOMWidgetImpl 在 legacy（Nodes 2.0 关）下，定位 DOM 控件时硬扣
  // (computedHeight - 4) - (2*margin - 4) = computedHeight - 2*margin —— 实测
  // margin=10，所以总扣减 = 4 + 16 = 20。computedHeight 必须多报 20 才能让
  // wrapper 等于内容自然高度 hh()，否则最后一段会露出节点底边（实测 legacy
  // "9 功能规划" 头部被截在紫底外）。★ 只影响 legacy：
  //   - computeSize = hhReport()：保证 litegraph 在 node.computeSize() 里调
  //     widget.computeSize(width) 把返回值写回 widget.computedHeight 时不会
  //     把 inflation 抹掉；
  //   - computeLayoutSize / CSS var = hh()：Vue 那边没这 20px 扣减，不能加，
  //     否则 Vue 的 wrapper 会多出 20px 空白。
  // ★ isLegacy **不能靠 class 判**：Nodes 2.0 的 DOM widget 容器同样叫
  //   "dom-widget"（前端 DomWidget.vue: class="dom-widget size-full"），
  //   用 contains("dom-widget") 会把 Vue 模式误判成 legacy，多报 20px ——
  //   而 Vue 那边没有这 20px 扣减，底边就多出一条空白。
  //   官方标志是 LiteGraph.vueNodesMode（LGraphNode._arrangeWidgets 里就是
  //   用它区分两种模式的）。拿不到时退回原来的 class 判定。
  const LG = (typeof LiteGraph !== "undefined" && LiteGraph)
    || (typeof window !== "undefined" && window.LiteGraph);
  const isVue = !!(LG && LG.vueNodesMode);
  const isLegacy = (LG && typeof LG.vueNodesMode === "boolean")
    ? !LG.vueNodesMode
    : !!root.parentElement?.classList?.contains("dom-widget");
  /* ★ Nodes 2.0「背景不跟随版块」的真身 —— 容器被 100% 高度钉死了：
   *   DomWidget.vue 的容器是 `class="dom-widget size-full"`，即 height:100%，
   *   这个 100% 是相对**父级 grid 行高**的，而行高来自 layout store 上一次
   *   记录的高度。于是形成死锁：面板内容长高 → 溢出容器 → 但容器高度是
   *   100%（旧值）不动 → 节点根元素高度不变 → useVueNodeResizeTracking 的
   *   共享 ResizeObserver 量不到变化 → layout store 不更新 → 背景停在旧高度，
   *   版块一路画到紫底外面。
   *   （对照 legacy：_arrangeWidgets 里 `if (!LiteGraph.vueNodesMode && y >
   *   bodyHeight) this.setSize(...)` —— Vue 模式下 litegraph **根本不 setSize**，
   *   背景高度只认 DOM，所以必须让 DOM 真的被撑开。）
   *   修法：容器高度回到 auto，并用 min-height 钉住内容高度。这样内容比
   *   分配到的行高更高时能把容器（进而把节点）撑开，RO 才量得到。
   *   legacy 下不动它 —— 那边容器高度由 litegraph 自己写，插手会打架。 */
  try {
    const wrapH = root.parentElement;
    if (wrapH) {
      if (isVue) {
        // ★ 只设 height:auto，不加 min-height：
        //   min-height = hh() 看起来是"保底"，实际上 hh() 包含了 textarea 的
        //   CSS min-height:260px 等所有内容元素的"最小值"，内容少时 hh() 就
        //   远超真实需要的高度 → 容器被锁在过大的最小值上 → 节点背景底部空
        //   出一大片（实测 textarea 13 行、root 实际 ~270，hh() 却算到 ~480，
        //   结果节点背景比内容高 ~220px，正是用户截图里的现象）。
        //   height:auto 本身已经能让容器由内容撑开（flex/grid item 的
        //   height:auto 会按内容算），且 ResizeObserver 会量到新高度反馈给
        //   layout store —— 上一轮担心"grid 行高不给内容撑开"是多余的，实测
        //   auto 即可。
        wrapH.style.height = "auto";
        wrapH.style.minHeight = "";
        /* align-self:start —— 不让 grid 把容器拉伸到行高，避免「显示高级
         * 输入」按钮下方出现节点背景空段（grid 行高 > 容器内容高度时容器被
         * 撑大、底部露白）。详情见坑9 沉淀。 */
        wrapH.style.alignSelf = "start";
      } else {
        wrapH.style.height = "";
        wrapH.style.minHeight = "";
        wrapH.style.alignSelf = "";
      }
    }
  } catch (e) { /* 拿不到就算了 */ }
  const hhReport = () => hh() + (isLegacy ? 20 : 0);
  // ★ 用户拖节点边缘时 litegraph 会调 dom.computeSize() 重新计算布局，
  //   但不会主动调 panelLayout。节点拉宽后 min-width 没跟上，紫底就只覆盖
  //   左半边。在这里 schedule 一次 panelLayout 异步重钉：
  //   - Promise.resolve() 让 panelLayout 在 litegraph 的写入之后跑
  //   - __layoutSync 节流：连调 100 次只刷一次
  dom.computeSize = (width) => {
    const w = Math.max(Number(width) || 0, 240);
    if (!dom.__layoutSync) {
      dom.__layoutSync = true;
      Promise.resolve().then(() => {
        dom.__layoutSync = false;
        try { panelLayout(dom, root, node); } catch (e) { /* 节点没了 */ }
      });
    }
    return [w, hhReport()];
  };
  try {
    dom.computeLayoutSize = () => {
      const v = hh();
      return { minHeight: v, maxHeight: v, minWidth: 0 };
    };
    dom.computedHeight = hhReport();
  } catch (e) { /* 老版本没有这套 */ }
  try {
    root.style.setProperty("--comfy-widget-height", hh() + "px");
    root.style.setProperty("--comfy-widget-min-height", hh() + "px");
    root.style.setProperty("--comfy-widget-max-height", hh() + "px");
  } catch (e) { /* 不支持自定义属性就算了 */ }
}

/* ---- 分辨率：复刻 comfy_extras/nodes_resolution.py 的 ResolutionSelector ---- */
const ASPECT_RATIOS = {
  "1:1 (Square)": [1, 1],
  "2:3 (Portrait Photo)": [2, 3],
  "3:2 (Photo)": [3, 2],
  "3:4 (Portrait Standard)": [3, 4],
  "4:3 (Standard)": [4, 3],
  "9:16 (Portrait Widescreen)": [9, 16],
  "16:9 (Widescreen)": [16, 9],
  "21:9 (Ultrawide)": [21, 9],
};

const DEFAULT_ASPECT = "16:9 (Widescreen)";

/** 与 ResolutionSelector.execute() 完全同一套算式，前端只是提前显示。 */
function calcResolution(aspect, megapixels, multiple) {
  const r = ASPECT_RATIOS[aspect] || ASPECT_RATIOS[DEFAULT_ASPECT];
  const m = Math.max(1, Math.round(Number(multiple) || 8));
  const scale = Math.sqrt(((Number(megapixels) || 1) * 1024 * 1024) / (r[0] * r[1]));
  return [Math.round((r[0] * scale) / m) * m, Math.round((r[1] * scale) / m) * m];
}

/** 从 PACK 的「画幅」文本猜比例："16:9 / 2K / 24fps" → "16:9 (Widescreen)"。 */
function guessAspect(text) {
  const m = /(\d{1,2})\s*[:：]\s*(\d{1,2})/.exec(String(text || ""));
  if (!m) return "";
  const want = m[1] + ":" + m[2];
  return Object.keys(ASPECT_RATIOS).find((k) => k.indexOf(want + " ") === 0) || "";
}

/* ------------------------------------------------------------------ */
/* 小马 SVG —— 用户提供的奔马剪影（dancinghorse.svg）                   */
/* ------------------------------------------------------------------ */
/* viewBox 是剪影的紧致包围框（无头 Chromium 量轮廓 + Pillow 扫 bbox：
 * x 115..907 / y 160..864，切缝 y0=604，剪影 bbox 见 PONY_LEGS 注释）。
 * 原图马头朝左，CSS 用 scaleX(-1) 翻成朝右（跟轨道前进方向一致）。
 *
 * ★ 单条 path 的整马剪影，腿和身子长在一起 —— 腿只能从剪影里「切」出来，
 *   绝不能另画直棍（位置/粗细/朝向对不上，会错位）。每条腿用一个矩形
 *   （PONY_LEGS 的 x0/x1/y0/y1）从身子掩膜挖掉，矩形必须盖过蹄子最低点
 *   （y1 给足），否则那截腿留在身子里、看着像多一块。
 * ★ 腿用**水平剪切**动（见 legShear），不用旋转 —— 绕切缝 y=y0 错切，
 *   切缝处位移恒为 0，腿和身子永远合缝，无尖刺/台阶。
 * ★ 摆动不用 CSS transform（transform-origin 各家解释不一），直接在 rAF
 *   里 setAttribute("transform", …)。 */
const PONY_VIEWBOX = "94 160 834 704";
const PONY_PATH = "M140.8 162.95936l25.76384 34.34496-2.53952-38.01088 27.0336 35.57376c155.83232-28.20096 209.77664 155.81184 209.77664 155.81184 80.97792 76.0832 202.4448 50.29888 202.4448 50.29888 100.63872-18.41152 153.31328 49.07008 153.31328 49.07008 103.03488 6.12352 107.99104 128.79872 107.99104 128.79872-1.26976 67.50208 44.15488 109.19936 44.15488 109.19936-100.61824 1.20832-117.78048-127.56992-117.78048-127.56992-8.6016-96.93184-27.01312-82.2272-27.01312-82.2272 19.68128 76.06272-77.29152 153.33376-77.29152 153.33376 3.70688 12.26752 20.93056 15.95392 20.93056 15.95392 46.57152 25.76384 19.61984 35.59424 19.61984 35.59424-15.93344 25.78432-39.30112 20.86912-39.30112 20.86912 2.43712 29.45024-29.45024 51.52768-29.45024 51.52768-18.39104 44.15488-57.61024 77.29152-57.61024 77.29152-7.35232 0-11.03872 8.6016-11.03872 8.6016 0 23.30624-30.69952 23.28576-30.69952 23.28576-20.91008 0-23.30624-6.12352-23.30624-6.12352 3.6864-30.67904 49.07008-39.23968 49.07008-39.23968 18.39104-13.49632 4.93568-17.2032 4.93568-17.2032-28.24192 7.35232-11.03872-31.90784-11.03872-31.90784 1.2288-46.592 39.2192-56.44288 39.2192-56.44288 15.93344-3.6864 17.2032-12.24704 17.2032-12.24704-24.576-20.86912-35.59424-87.1424-35.59424-87.1424-14.72512 2.43712-35.5328-13.47584-35.5328-13.47584-74.87488 39.23968-219.62752-8.6016-219.62752-8.6016 0 35.55328-18.41152 83.43552-18.41152 83.43552 8.56064 87.1424-2.4576 117.78048-2.4576 117.78048-9.8304 22.07744-11.03872 20.84864-11.03872 20.84864 18.37056 24.53504-7.3728 28.22144-7.3728 28.22144h-23.28576c-39.28064-2.47808-14.72512-22.07744-14.72512-22.07744 37.9904-24.55552 33.11616-56.4224 33.11616-56.4224-1.20832-19.61984-6.16448-110.42816-6.16448-110.42816-2.4576-39.23968-15.91296-51.54816-15.91296-51.54816-13.5168 18.41152-93.22496 26.97216-93.22496 26.97216-25.76384 45.38368 34.32448 88.33024 34.32448 88.33024 11.0592-19.64032 28.2624 3.6864 28.2624 3.6864 17.12128 35.57376 6.08256 36.80256 6.08256 36.80256-14.72512 1.20832-39.2192-20.84864-39.2192-20.84864-46.63296-9.80992-71.20896-93.26592-71.20896-93.26592-14.72512-45.40416 56.48384-56.4224 56.48384-56.4224 58.90048-22.07744 36.80256-40.48896 36.80256-40.48896-33.13664-47.84128-6.144-76.0832-6.144-76.0832 13.5168-14.72512 12.24704-42.94656 12.24704-42.94656-24.55552-52.75648-23.30624-117.76-23.30624-117.76-7.33184 26.99264-40.48896 47.84128-40.48896 47.84128s7.3728-46.61248 0 4.9152c-7.33184 51.52768-49.0496 24.51456-49.0496 24.51456-29.45024-18.39104-4.95616-63.7952-4.95616-63.7952-1.29024-50.31936 24.55552-107.95008 24.55552-107.95008l-8.54016-52.67456z";

/* 腿用水平剪切（shear）而非旋转：绕切缝 y=y0 做 x' = x + s·(y−y0)，
 * 切缝处位移恒为 0 → 腿与身子永不合缝。SVG 写法 matrix(1 0 s 1 (−s·y0) 0)，
 * s = −tan(θ)（马头朝左，往左推取负）。矩形需比腿宽且 y1 盖过蹄子最低点，
 * 否则腿边留碎边 / 蹄子被截一截留在身子里。三条腿矩形见 PONY_LEGS。 */
const PONY_LEGS = [
  { x0: 138, x1: 277, y0: 604, y1: 766, base: 0, amp:  13, phase: 0.00 }, // 抬起的前腿
  { x0: 277, x1: 348, y0: 604, y1: 850, base: 0, amp:  13, phase: 0.50 }, // 着地的前腿
  { x0: 530, x1: 742, y0: 604, y1: 872, base: 0, amp: -13, phase: 0.25 }, // 后腿
];
/* 步态周期。摆幅要按**显示尺寸**倒推：viewBox 834 宽塞进 32px，
 * 1 个单位 ≈ 0.038px —— 所以 1px 的视觉位移要 26 个单位，别按直觉给小数。
 * 13° 的剪切让蹄子横扫 ≈ tan13° × 140 ≈ 32 单位 ≈ 1.2px。 */
const PONY_GAIT_MS = 430;
/* 整体起伏：bob 上下颠（单位）、pitch 绕肚线俯仰（度） */
const PONY_BOB = 26;
const PONY_PITCH = 2.6;
const PONY_PITCH_AT = [511, 512];
/* 腿的裁剪矩形往上多留这么多单位，压住身子的切边、消掉抗锯齿细线 */
const PONY_CUT_OVERLAP = 3;
let _ponyMaskSeq = 0;

/** 建一个矩形元素（掩膜里当"白底全留 / 黑块全挖"用） */
function svgRect(NS, x, y, w, hh, fill) {
  const r = document.createElementNS(NS, "rect");
  r.setAttribute("x", x); r.setAttribute("y", y);
  r.setAttribute("width", w); r.setAttribute("height", hh);
  r.setAttribute("fill", fill);
  return r;
}

/** 一条腿的剪切矩阵：绕 y=y0 做水平错切，角度 a（度，正=朝马头方向） */
function legShear(L, a) {
  const s = -Math.tan(a * Math.PI / 180);
  return "matrix(1 0 " + s.toFixed(5) + " 1 " + (-s * L.y0).toFixed(2) + " 0)";
}

/** 建一张 userSpaceOnUse 的掩膜，画布盖住整条 viewBox 并留出剪切位移的余量。
 *  余量是必须的：腿剪切后下端会横移（13° ≈ 60 单位），掩膜画布太窄会把
 *  蹄子切掉。这里取 x 60..960 / y 140..880，包住剪影 115..907 × 160..864
 *  以及剪切后的极限位置。 */
function svgMask(NS, id) {
  const m = document.createElementNS(NS, "mask");
  m.setAttribute("id", id);
  m.setAttribute("maskUnits", "userSpaceOnUse");
  m.setAttribute("x", "60"); m.setAttribute("y", "140");
  m.setAttribute("width", "900"); m.setAttribute("height", "740");
  return m;
}

function ponySvg() {
  const NS = "http://www.w3.org/2000/svg";
  const FILL = "#afcd50";
  const wrap = h("div", "h3d-pony");
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", PONY_VIEWBOX);

  const seq = ++_ponyMaskSeq;
  const defs = document.createElementNS(NS, "defs");

  // 身子：整条剪影，把每条腿的矩形挖掉（腿的位置就空了）
  // 白底 rect 必须**和掩膜画布一样大**，否则身子会被自己掩膜的边缘切掉。
  const bodyMaskId = "h3d-body-" + seq;
  const bm = svgMask(NS, bodyMaskId);
  bm.appendChild(svgRect(NS, 60, 140, 900, 740, "#fff"));
  PONY_LEGS.forEach((L) => {
    bm.appendChild(svgRect(NS, L.x0, L.y0, L.x1 - L.x0, L.y1 - L.y0, "#000"));
  });
  defs.appendChild(bm);

  // 每条腿一张掩膜：矩形比身子那边的切口**往上多留 3 单位**，
  // 让腿压住身子的切边 —— 两条同色路径严格贴边时抗锯齿会留一条细白线，
  // 叠一点就没了。3 单位在剪切下只横移 0.7 单位，看不出来。
  const rig = PONY_LEGS.map((L, i) => {
    const rectId = "h3d-leg-" + seq + "-" + i;
    const rm = svgMask(NS, rectId);
    rm.appendChild(svgRect(NS, L.x0, L.y0 - PONY_CUT_OVERLAP,
                           L.x1 - L.x0, L.y1 - L.y0 + PONY_CUT_OVERLAP, "#fff"));
    defs.appendChild(rm);
    return { rectId };
  });
  svg.appendChild(defs);

  // bob：整体起伏的容器；腿放它里面，才会跟着身子一起颠
  const bob = document.createElementNS(NS, "g");

  const body = document.createElementNS(NS, "path");
  body.setAttribute("fill", FILL);
  body.setAttribute("d", PONY_PATH);
  body.setAttribute("mask", "url(#" + bodyMaskId + ")");
  bob.appendChild(body);

  // 腿：剪影 ∩ 矩形，整组做水平剪切。剪切不移动切缝（y=y0 处位移恒为 0）
  // → 不管腿怎么动，和身子的接缝都严丝合缝，永远不会露出缝或尖刺。
  const legs = rig.map((D, i) => {
    const g = document.createElementNS(NS, "g");
    const p = document.createElementNS(NS, "path");
    p.setAttribute("fill", FILL);
    p.setAttribute("d", PONY_PATH);
    p.setAttribute("mask", "url(#" + D.rectId + ")");
    g.appendChild(p);
    g.setAttribute("transform", legShear(PONY_LEGS[i], 0));
    bob.appendChild(g);
    return { g, i };
  });

  svg.appendChild(bob);
  wrap.appendChild(svg);
  return { wrap, svg, bob, legs };
}

/** 按时间把身子和三条腿摆到位（ms 是自起跑起的毫秒数） */
function ponyGait(pony, ms) {
  const t = (ms % PONY_GAIT_MS) / PONY_GAIT_MS;   // 0..1
  const w = t * Math.PI * 2;
  if (pony.bob) {
    pony.bob.setAttribute("transform",
      "translate(0 " + (Math.sin(w) * PONY_BOB).toFixed(1) + ") "
      + "rotate(" + (Math.sin(w + 0.7) * PONY_PITCH).toFixed(2)
      + " " + PONY_PITCH_AT[0] + " " + PONY_PITCH_AT[1] + ")");
  }
  (pony.legs || []).forEach((leg, i) => {
    const L = PONY_LEGS[i];
    if (!L) return;
    const a = L.base + L.amp * Math.sin((t + L.phase) * Math.PI * 2);
    leg.g.setAttribute("transform", legShear(L, a));
  });
}

/* ------------------------------------------------------------------ */
/* 版块：节点参数 —— 把 H3Director 原生 widget 全部代理成统一 DOM 控件。
 *  原生控件排在 DOM 面板前会把面板顶到下半截，故收起（hideWidget）后由面板
 *  代理；读写走 setWidget，值仍是原生 widget 的值，存盘/加载/连线不变。
 *  设一次就不动的两组（渲染设置/输出选项）默认折叠。                      */
/* ------------------------------------------------------------------ */
/** kind: sel=下拉 / txt=单行 / num=数字 / area=多行 / chk=勾选
 *  fold: true = 默认折叠。下列 SILENT_WIDGETS 参数不进面板（只留值），
 *  避免"面板一份 + 节点原生一份"重复；如需恢复，从 SILENT_WIDGETS 摘掉并
 *  在 fields 加回即可。 */
const PARAM_SECTIONS = [
  {
    title: "提示词",
    note: "直接读写节点上的原生参数，存盘 / 队列 / 连线照旧。",
    fields: [
      { name: "global_prompt", label: "全局提示词", kind: "area",
        ph: "追加到每个分镜提示词后面（整体风格 / 声音氛围）" },
      { name: "prompt_lines", label: "分镜提示词", kind: "area",
        ph: "仅「源视频自动切分」生效：一行一个镜头" },
    ],
  },
  {
    title: "分镜切分",
    note: "决定脚本怎么切成镜头；改完要重新「解析 PACK」才生效。",
    fields: [
      { name: "sensitivity", label: "切点灵敏度", kind: "sel" },
      { name: "min_shot_seconds", label: "最短镜头", kind: "num",
        unit: "秒", min: 0.1, step: 0.1 },
      { name: "max_shot_seconds", label: "最长镜头", kind: "num",
        unit: "秒", min: 0.1, step: 0.1 },
      { name: "max_shots", label: "最多分镜", kind: "num",
        unit: "段", min: 0, step: 1, ph: "0 = 全部" },
      { name: "handoff_seconds", label: "回放衔接", kind: "num",
        unit: "秒", min: 0, step: 0.05 },
      { name: "fallback_prompt", label: "缺省提示词", kind: "area" },
    ],
  },
  {
    title: "渲染设置",
    fold: true,
    note: "画质与显存。步数 / 调度器 / 闪电渲染已收起，默认跟随上游采样器。",
    fields: [
      { name: "crf", label: "成片质量", kind: "num",
        min: 0, max: 51, step: 1, ph: "越小越清晰" },
      { name: "stabilize", label: "亮度稳定", kind: "num",
        min: 0, max: 1, step: 0.05 },
      { name: "unload_models_after", label: "每段后卸载模型", kind: "chk" },
      { name: "resume", label: "断点续接", kind: "chk" },
      { name: "duration_is_new_content", label: "duration = 新增时长", kind: "chk" },
    ],
  },
  {
    title: "输出选项",
    fold: true,
    note: "成片之外的产物开关。",
    fields: [
      { name: "build_timeline", label: "生成时间线", kind: "chk" },
      { name: "export_segments", label: "逐段导出 MP4", kind: "chk" },
      { name: "ref_image_size", label: "参考图尺寸", kind: "sel" },
    ],
  },
];

/** 纯数据字段 + 删繁就简后收起的参数：不露出来，也不让它们把面板顶下去。
 *
 *  前半是本来就不该见的（连线进来的 JSON / PACK / 面板自己写回的分类）；
 *  后半是用户点名要删的 9 个 —— 它们在面板里有代理、节点上又有原生控件，
 *  两套并存就是「重复项」。值都还在（存盘照旧），只是不再占版面。 */
const SILENT_WIDGETS = [
  // 纯数据：由连线 / 面板自己写入，不需要人改
  "shots_json", "pack_info", "ref_classify",
  // 纯内部驱动：只被 repairSegment() 用 setWidget() 直接改值，面板里没有
  // 对应的代理控件，用户也不该手填——之前漏掉了，非 Nodes 2.0 下就是「单段
  // 修复 (0=关闭)」这个原生控件裸露在节点上的来源。
  "repair_segment",
];

/** 所有该被隐藏的原生控件名 = 参数面板已经做了代理的 + 纯静默的。
 *  给 mountPanels 里的定时重钉用，对抗 Nodes 2.0 下 Vue 重渲染把隐藏
 *  样式覆盖回去的问题。 */
const ALL_HIDDEN_WIDGET_NAMES = PARAM_SECTIONS
  .flatMap((spec) => spec.fields.map((f) => f.name))
  .concat(SILENT_WIDGETS);

function buildWidgetSection(node, spec) {
  const sec = h("div", "h3d-sec"
    + (spec.title === "提示词" ? " h3d-sec--prompt" : ""));
  const head = sectionHead(0, spec.title);      // 序号由 mountPanels 统一发
  const hd = head.hd, tag = head.tag;
  const btnSync = h("button", "h3d-btn", "刷新");
  hd.appendChild(btnSync);
  const bd = h("div", "h3d-bd");
  if (spec.fold) bd.classList.add("collapsed");

  const box = h("div", "h3d-fields");
  bd.appendChild(box);
  if (spec.note) bd.appendChild(h("div", "h3d-note", spec.note));

  const ctrls = [];

  spec.fields.forEach((f) => {
    const w = findWidget(node, f.name);
    if (!w) return;                                  // 这版没有该参数就跳过
    const row = h("div", "h3d-field");

    if (f.kind === "chk") {
      const el = document.createElement("input");
      el.type = "checkbox";
      const lab = h("label", "h3d-chk");
      lab.appendChild(el);
      lab.appendChild(document.createTextNode(f.label));
      row.appendChild(lab);
      el.checked = !!w.value;
      el.onchange = () => setWidget(node, f.name, el.checked);
      ctrls.push({ w, write: () => { el.checked = !!w.value; } });
      box.appendChild(row);
      return;
    }

    row.appendChild(h("span", "lb", f.label));
    let el;
    if (f.kind === "sel") {
      const raw = w.options && (w.options.values || w.options);
      const opts = Array.isArray(raw) ? raw : [];
      if (opts.length) {
        el = document.createElement("select");
        el.className = "h3d-in is-sel";
        opts.forEach((o) => {
          const op = document.createElement("option");
          op.value = String(o);
          op.textContent = String(o);
          el.appendChild(op);
        });
        el.value = w.value == null ? "" : String(w.value);
        el.onchange = () => setWidget(node, f.name, el.value);
      } else {
        // 拿不到选项就别做成死下拉 —— 退化成文本框，至少还改得动
        el = document.createElement("input");
        el.className = "h3d-in";
        el.type = "text";
        el.value = w.value == null ? "" : String(w.value);
        const push = () => setWidget(node, f.name, el.value);
        el.oninput = push; el.onchange = push;
      }
    } else if (f.kind === "num") {
      el = document.createElement("input");
      el.className = "h3d-in is-num";
      el.type = "number";
      if (f.min != null) el.min = String(f.min);
      if (f.max != null) el.max = String(f.max);
      el.step = String(f.step != null ? f.step : 0.1);
      el.value = w.value == null ? "" : String(w.value);
      const push = () => {
        const n = parseFloat(el.value);
        if (isFinite(n)) setWidget(node, f.name, n);
      };
      el.oninput = push; el.onchange = push;
    } else {
      el = document.createElement(f.kind === "area" ? "textarea" : "input");
      el.className = "h3d-in";
      if (f.kind === "area") row.classList.add("is-block");
      else el.type = "text";
      if (f.ph) el.placeholder = f.ph;
      el.value = w.value == null ? "" : String(w.value);
      const push = () => setWidget(node, f.name, el.value);
      el.oninput = push;
      el.onchange = push;
      // B2：仅为 prompt_lines 挂 @ 提及菜单（<Picture N> 是这里引用的约定）
      // B2：仅为 prompt_lines 挂 @ 提及菜单 + 胶囊预览条（<Picture N> 是这里引用的约定）
      if (f.name === "prompt_lines") el.__h3TokView = attachMentionMenu(el, node);
    }

    if (f.kind === "num" && f.unit) {
      const wrap = h("span", "h3d-numwrap");
      wrap.appendChild(el);
      wrap.appendChild(h("span", "unit", f.unit));
      row.appendChild(wrap);
    } else {
      row.appendChild(el);
      // 胶囊预览条紧跟在 textarea 下面（prompt_lines 才有）
      if (el.__h3TokView) row.appendChild(el.__h3TokView);
    }

    ctrls.push({
      w,
      write: () => {
        const v = w.value == null ? "" : String(w.value);
        if (isFocused(el)) return;                   // 别打断正在输入的人
        if (el.value !== v) {
          el.value = v;
          // 值被外部改写（载入 PACK / 切换会话）不会触发 input 事件，得手动重画胶囊
          if (el.__h3TokView) refreshTokenViews();
        }
      },
    });
    box.appendChild(row);
  });

  // 这个节点上没有任何一个字段 → 整个版块没必要存在（别留空壳）
  if (!ctrls.length) return null;

  collapsible(node, bd, head.title, head.badge);

  function render() {
    ctrls.forEach((c) => c.write());
    // 面板确实建出来了才收原生控件 —— 万一构建失败，原生还在，用户不至于没得改
    spec.fields.forEach((f) => hideWidget(node, f.name));
    tag.textContent = ctrls.length + " 项";
    tag.className = "h3d-tag";
    refit(node);
  }
  btnSync.onclick = (e) => { e.stopPropagation(); render(); };
  sec.appendChild(hd);
  sec.appendChild(bd);
  return { sec, badge: head.badge, render };
}

/* ------------------------------------------------------------------ */
/* 版块 2：剧本 PACK —— 只 5 个字段，一行紧凑键值                       */
/* ------------------------------------------------------------------ */
function buildPackPanel(node, state) {
  const sec = h("div", "h3d-sec");
  const head = sectionHead(0, "剧本 PACK");
  const hd = head.hd, tag = head.tag;
  tag.textContent = "未解析";
  const btn = h("button", "h3d-btn is-primary", "解析 PACK");
  hd.appendChild(btn);
  const bd = h("div", "h3d-bd");

  const meta = h("div", "h3d-meta");
  const cells = {};
  [
    ["项目", "project", "—"],
    ["模式", "mode", "—"],
    ["总时长", "total_duration", "—"],
    ["段数", "segments", "—"],
    ["画幅", "aspect", "—"],
  ].forEach(([label, key, fallback]) => {
    const it = h("div", "h3d-item");
    it.appendChild(h("span", "k", label));
    const v = h("span", "v is-empty", fallback);
    cells[key] = v;
    it.appendChild(v);
    meta.appendChild(it);
  });
  bd.appendChild(meta);

  /* ★ 第 30 轮：后端 /h3/pack_preview 算出的 issues 以前整条丢弃 ——
     老代码只把头部 5 项填进 meta，issues 拿到手就扔了。
     后果很实在：PACK 里某段正文被清空时（实测 —— 工作流 JSON 里 PACK 存了
     两份 widgets_values / widgets_values_named，ComfyUI 加载了 S01 英文块
     被清空的那一份），界面上一个字都不报，用户只看到空提示词框，
     分不清是"本来就没内容"还是"数据坏了"。
     这里补一条诊断区：普通 issue 黄字，致命 issue（整段正文为空）红字，
     并且右上「N 段」徽标一起变红 —— 让坏数据自己会喊。 */
  const issueBox = h("div", "h3d-note is-warn");
  issueBox.style.display = "none";
  issueBox.style.marginTop = "6px";
  bd.appendChild(issueBox);

  collapsible(node, bd, head.title, head.badge);

  async function parse() {
    const text = state.packText();
    if (!text || !text.trim()) {
      tag.textContent = "无 PACK";
      tag.className = "h3d-tag is-bad";
      state.onPack(null);
      return;
    }
    tag.textContent = "解析中…";
    tag.className = "h3d-tag";
    try {
      // 顺手把画布上 H3PromptPackParser 的 session_name 控件值带过去：
      // 后端算会话名时「控件 > PACK 头部 Project」这个顺序必须和那个节点
      // 一致，否则面板查的会话和落盘的会话会是两个名字。
      const pkNode = findNodeByClass("H3PromptPackParser");
      const pkSess = readStr(pkNode, "session_name", "");
      const res = await api.fetchApi("/h3/pack_preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, session_name: pkSess }),
      });
      const data = await res.json();
      if (!data.ok) throw new Error(data.error || "解析失败");
      const head = data.header || {};

      // 显示 5 项头部信息；后端 issues 现在也要显示（见 issueBox 注释）
      ["project", "mode", "total_duration", "segments", "aspect"]
        .forEach((key) => {
          const el = cells[key];
          if (!el) return;
          let raw = key === "segments"
            ? (head.segments || String(data.segments.length))
            : head[key];
          if (!raw) {
            el.textContent = "—";
            el.className = "v is-empty";
            return;
          }
          el.textContent = "";
          el.className = "v";
          el.appendChild(document.createTextNode(String(raw)));
          if (key === "total_duration") {
            const tn = data.total_new_seconds, tg = data.total_gen_seconds;
            if (tn != null && tg != null) {
              const sub = h("span", "sub");
              sub.textContent = "  " + tn.toFixed(1) + "s 新增 / "
                                + tg.toFixed(1) + "s 生成";
              el.appendChild(sub);
            }
          }
        });

      // ★ 第 30 轮：把后端诊断真正显示出来（以前整条丢弃，见 issueBox 注释）。
      //   致命 issue（整段正文为空）转红字，徽标也转红 —— 一眼看见坏数据。
      const issues = Array.isArray(data.issues) ? data.issues : [];
      const fatal = issues.filter((s) => H3_FATAL_ISSUE_RE.test(String(s)));
      issueBox.replaceChildren();
      if (!issues.length) {
        issueBox.style.display = "none";
      } else {
        issueBox.style.display = "";
        issueBox.className = fatal.length
          ? "h3d-note is-warn is-fatal" : "h3d-note is-warn";
        issueBox.appendChild(document.createTextNode(
          (fatal.length ? "⚠ " : "") + issues.join("   ·   ")));
      }

      tag.textContent = data.segments.length + " 段"
        + (fatal.length ? " ⚠" : "");
      tag.className = fatal.length ? "h3d-tag is-bad" : "h3d-tag is-ok";
      state.onPack(data);
    } catch (err) {
      tag.textContent = "解析失败";
      tag.className = "h3d-tag is-bad";
      issueBox.style.display = "none";
      state.onPack(null);
    }
    refit(node);   // PACK 解析完，项目信息那行会变宽，节点要跟着贴合
  }
  btn.onclick = (e) => { e.stopPropagation(); parse(); };

  sec.appendChild(hd);
  sec.appendChild(bd);
  return { sec, badge: head.badge, parse };
}

/* ------------------------------------------------------------------ */
/* 版块 2：输出规格 —— 分辨率（画幅比 + 像素量）+ 时长（代理上游节点）   */
/* ------------------------------------------------------------------ */
/* 分辨率读/写 ResolutionSelector（无则直写 H3 Chain Settings）；时长读/写
 * PACK 解析器的 default_duration 与 H3Director 的 handoff_seconds。 */
function buildOutputPanel(node, state) {
  const sec = h("div", "h3d-sec");
  const head = sectionHead(0, "输出规格");
  const hd = head.hd, tag = head.tag;
  const btnSync = h("button", "h3d-btn", "同步");
  hd.appendChild(btnSync);
  const bd = h("div", "h3d-bd");

  const box = h("div", "h3d-fields");
  bd.appendChild(box);

  const resRow = h("div", "h3d-field");
  const durRow = h("div", "h3d-field");
  const noteRes = h("div", "h3d-note", "");
  const noteDur = h("div", "h3d-note", "");

  const csNode = () => findNodeByClass("H3ChainSettings");
  const pkNode = () => findNodeByClass("H3PromptPackParser");
  /** 图上可能有多个 ResolutionSelector；优先取真正接到画布上的那一个。 */
  const rsNode = () => {
    const all = graphNodes().filter((n) => nodeClass(n) === "ResolutionSelector");
    if (all.length <= 1) return all[0] || null;
    const c = csNode();
    if (c) {
      const feed = all.find((r) => {
        try { return upstream(c, "width") === r; } catch (e) { return false; }
      });
      if (feed) return feed;
    }
    return all[0];
  };

  let mode = "";
  let selAspect = null, inMp = null, outRes = null;   // 模式 A：ResolutionSelector
  let inW = null, inH = null;                         // 模式 B：直接写画布宽高
  let inDur = null, inHand = null;                    // 时长

  /* ---- 分辨率控件：按图上有没有 ResolutionSelector 决定形态 ---- */
  /** ResolutionSelector 是否真的在驱动画布：它的 width 输出有没有接到 Chain Settings。 */
  function selectorFeedsChain(r, c) {
    if (!r || !c) return false;
    try { return upstream(c, "width") === r; } catch (e) { return false; }
  }

  function detectMode() {
    const r = rsNode(), c = csNode();
    if (!c) return r ? "selector" : "none";
    return selectorFeedsChain(r, c) ? "selector" : "direct";
  }

  function buildRes() {
    const next = detectMode();
    if (next === mode) return;
    mode = next;
    resRow.innerHTML = "";
    resRow.appendChild(h("span", "lb", "分辨率"));
    selAspect = inMp = outRes = inW = inH = null;

    if (next === "selector") {
      selAspect = document.createElement("select");
      selAspect.className = "h3d-in is-sel";
      Object.keys(ASPECT_RATIOS).forEach((k) => {
        const o = document.createElement("option");
        o.value = k;
        o.textContent = k;
        selAspect.appendChild(o);
      });
      resRow.appendChild(selAspect);

      inMp = document.createElement("input");
      inMp.type = "number";
      inMp.className = "h3d-in is-num";
      inMp.min = "0.1"; inMp.max = "16"; inMp.step = "0.1";
      inMp.title = "目标总像素量（ResolutionSelector 的 megapixels）";
      resRow.appendChild(inMp);
      resRow.appendChild(h("span", "unit", "MP"));
      resRow.appendChild(h("span", "arrow", "→"));
      outRes = h("span", "h3d-out is-mute", "— × —");
      resRow.appendChild(outRes);

      selAspect.onchange = pushRes;
      inMp.onchange = pushRes;
    } else if (next === "direct") {
      inW = document.createElement("input");
      inW.type = "number";
      inW.className = "h3d-in is-num";
      inW.min = "32"; inW.step = "32";
      resRow.appendChild(inW);
      resRow.appendChild(h("span", "unit", "×"));
      inH = document.createElement("input");
      inH.type = "number";
      inH.className = "h3d-in is-num";
      inH.min = "32"; inH.step = "32";
      resRow.appendChild(inH);
      resRow.appendChild(h("span", "unit", "px"));
      inW.onchange = pushRes;
      inH.onchange = pushRes;
    } else {
      resRow.appendChild(h("span", "h3d-out is-mute", "图上没有分辨率节点"));
    }
  }

  /* ---- 时长控件 ---- */
  (function buildDur() {
    durRow.appendChild(h("span", "lb", "时长"));
    inDur = document.createElement("input");
    inDur.type = "number";
    inDur.className = "h3d-in is-num";
    inDur.min = "0.25"; inDur.max = "15"; inDur.step = "0.25";
    inDur.title = "PACK 段标题没写时长时的缺省段长（秒）";
    durRow.appendChild(inDur);
    durRow.appendChild(h("span", "unit", "s 缺省段长"));

    inHand = document.createElement("input");
    inHand.type = "number";
    inHand.className = "h3d-in is-num";
    inHand.min = "0"; inHand.max = "4"; inHand.step = "0.125";
    inHand.title = "本段结尾回放给下一段的秒数（H3Director 的回放衔接）";
    durRow.appendChild(inHand);
    durRow.appendChild(h("span", "unit", "s 回放衔接"));

    inDur.onchange = () => {
      const p = pkNode();
      if (p) setWidget(p, "default_duration", Math.max(0.25, Number(inDur.value) || 10));
      render();
    };
    inHand.onchange = () => {
      setWidget(node, "handoff_seconds", Math.max(0, Number(inHand.value) || 0));
      render();
    };
  })();

  box.appendChild(resRow);
  box.appendChild(noteRes);
  box.appendChild(durRow);
  box.appendChild(noteDur);

  /* ---- 把面板上的值写回图上的节点 ---- */
  function pushRes() {
    const r = rsNode(), c = csNode();
    if (mode === "selector" && r) {
      const aspect = selAspect.value || DEFAULT_ASPECT;
      const mp = Math.min(16, Math.max(0.1, Number(inMp.value) || 0.5));
      setWidget(r, "aspect_ratio", aspect);
      setWidget(r, "megapixels", mp);
      const [w, hh] = calcResolution(aspect, mp, readNum(r, "multiple", 8));
      // 顺手同步到 H3 Chain Settings：那边是连线驱动，值本来就等于算式结果，
      // 同步一下只是让节点上显示的宽高不再停留在旧数字上。
      if (c) { setWidget(c, "width", w); setWidget(c, "height", hh); }
    } else if (mode === "direct" && c) {
      const snap = (v, d) => Math.max(32, Math.round((parseInt(v, 10) || d) / 32) * 32);
      setWidget(c, "width", snap(inW.value, 960));
      setWidget(c, "height", snap(inH.value, 544));
    }
    render();
  }

  function render() {
    buildRes();
    const r = rsNode(), c = csNode(), p = pkNode();

    // ---- 分辨率 ----
    if (mode === "selector" && r) {
      const aspect = readStr(r, "aspect_ratio", DEFAULT_ASPECT);
      const mp = readNum(r, "megapixels", 1);
      const mul = readNum(r, "multiple", 8);
      if (!isFocused(selAspect)) selAspect.value = aspect;
      if (!isFocused(inMp)) inMp.value = String(mp);
      const [w, hh] = calcResolution(aspect, mp, mul);
      // 画布节点上残留的旧值（比如从别的工作流抄过来的尺寸）跟算式不一致时
      // 对齐一次，免得"节点上写着一个数、实际渲出来另一个数"。
      if (c) {
        if (readNum(c, "width", 0) !== w) setWidget(c, "width", w);
        if (readNum(c, "height", 0) !== hh) setWidget(c, "height", hh);
      }
      outRes.textContent = w + " × " + hh;
      outRes.className = "h3d-out";
      outRes.title = "ResolutionSelector：" + aspect + " · " + mp
        + " MP · 对齐 " + mul + "px";

      // 与 PACK 声明的画幅对账
      const packRaw = String((state.packData?.header?.aspect) || "").trim();
      const hit = guessAspect(packRaw);
      let note = "对齐 " + mul + "px · 0.1–16 MP，非标比例不提供";
      if (packRaw) {
        note = "PACK 声明 " + packRaw.split("/")[0].trim()
          + (hit ? (hit === aspect ? " ✓ 一致" : " ✗ 与当前不一致") : "")
          + " · 对齐 " + mul + "px";
      }
      noteRes.textContent = note;
      const bad = !!(hit && hit !== aspect);
      noteRes.className = "h3d-note" + (bad ? " is-warn is-link" : "");
      noteRes.onclick = bad ? () => { selAspect.value = hit; pushRes(); } : null;
      noteRes.title = bad ? "点一下按 PACK 的画幅对齐" : "";
      tag.textContent = w + "×" + hh;
      tag.className = "h3d-tag is-ok";
    } else if (mode === "direct" && c) {
      const w = readNum(c, "width", 0), hh = readNum(c, "height", 0);
      if (!isFocused(inW)) inW.value = w ? String(w) : "";
      if (!isFocused(inH)) inH.value = hh ? String(hh) : "";
      noteRes.textContent = r
        ? "画布由 H3 Chain Settings 自己决定（ResolutionSelector 没接到它上面）"
        : "图上没有 ResolutionSelector，直接写 H3 Chain Settings 的宽高（自动对齐 32）";
      noteRes.className = "h3d-note";
      tag.textContent = w && hh ? w + "×" + hh : "";
      tag.className = "h3d-tag " + (w && hh ? "is-ok" : "");
    } else {
      noteRes.textContent = "图上没有 ResolutionSelector，也没有 H3 Chain Settings";
      noteRes.className = "h3d-note is-warn";
      tag.textContent = "";
      tag.className = "h3d-tag";
    }

    // ---- 时长 ----
    if (p) {
      inDur.disabled = false;
      if (!isFocused(inDur)) inDur.value = String(readNum(p, "default_duration", 10));
    } else {
      inDur.disabled = true;
      inDur.value = "";
    }
    if (!isFocused(inHand)) inHand.value = String(readNum(node, "handoff_seconds", 1.625));

    const segs = (state.packData && state.packData.segments) || [];
    if (segs.length) {
      const list = segs.map((s) => (Number(s.new_seconds) || 0).toFixed(1)).join(" / ");
      noteDur.textContent = "实际每段新增：" + list + " s —— 单段时长改 PACK 段标题的 "
        + "「10s+1.6=11.6」即可；没写才用上面的「缺省段长」";
    } else {
      noteDur.textContent = "PACK 段标题写明了时长的（如 10s+1.6=11.6）以 PACK 为准；没写才用「缺省段长」。";
    }
    noteDur.className = "h3d-note";

    refit(node);   // 分辨率/时长行数会随模式变，节点高度跟着贴合
  }

  btnSync.onclick = (e) => { e.stopPropagation(); render(); };
  collapsible(node, bd, head.title, head.badge);

  sec.appendChild(hd);
  sec.appendChild(bd);
  return { sec, badge: head.badge, render };
}

/* ------------------------------------------------------------------ */
/* 版块 3：参考图 —— 缩略图网格，可替换 / 添加 / 移除                  */
/* ------------------------------------------------------------------ */
function buildRefPanel(node, state) {
  const sec = h("div", "h3d-sec");
  const head = sectionHead(0, "参考图");
  const hd = head.hd, tag = head.tag;
  const btnAdd = h("button", "h3d-btn", "添加图片");
  btnAdd.title = "上传一张新图，接到下一个空着的参考图槽位";
  hd.appendChild(btnAdd);

  /* 批量改分类（第 14 轮）：进入多选模式，勾选多张后一次性应用同一分类 */
  const btnBulk = h("button", "h3d-btn", "批量改分类");
  btnBulk.title = "勾选多张参考图，一次性把它们设成同一个分类（ESC 退出）";
  btnBulk.onclick = () => { if (bulk.on) exitBulk(); else enterBulk(); };
  hd.appendChild(btnBulk);

  /* 顶部统计（第 14 轮）：挂在 tag 里 —— tag 的位置正好是标题右侧，
   * 而 tag 原本只放「需 1/2/3/4」那一坨。现在拆成两个语义明确的部分：
   *   [已接 N / 需 M]  主信息，中性灰
   *   [缺 1/2/3/4]     警示，黄底，只在真缺时出现
   * 用固定子节点 + 只改 textContent，避免每次刷新重建 DOM。
   * ⚠ 不用 insertBefore：离线测试桩的假 DOM 没实现它，会 TypeError。 */
  const statEl = h("span", "h3d-refstat");
  const statNum = h("span", "num", "0");
  const statSep = h("span", "sep", "/");
  const statNeed = h("span", "num", "0");
  statEl.appendChild(statNum);
  statEl.appendChild(statSep);
  statEl.appendChild(statNeed);
  statEl.title = "已接进画布的参考图数 / 剧本里引用的 <Picture N> 总数";
  const missEl = h("span", "miss", "");
  statEl.appendChild(missEl);
  tag.appendChild(statEl);
  // 「选已有」按钮已按用户要求移除（2026-09-16 第 11 轮）。
  // 面板头部只保留「添加图片」一个入口，换已有的图请直接改画布上的 LoadImage。
  const bd = h("div", "h3d-bd");

  const grid = h("div", "h3d-refs");
  bd.appendChild(grid);

  const noteRef = h("div", "h3d-note", "");
  bd.appendChild(noteRef);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.className = "h3d-file";
  bd.appendChild(fileInput);
  let _targetSlot = null;   // null = 追加到新槽；数字 = 替换该槽

  /* ---- 批量改分类（2026-09-16 第 14 轮）----
   * 场景：4 张参考图里 3 张都是角色，一张张点太累。
   * 交互：顶部「批量改分类」→ 进入多选模式（卡片浮出勾选圈、hover 工具条让位）
   *      → 勾选若干张 → 底部应用条选一个分类 → 一次性写入并退出。
   * 退出三条路：按 ESC / 点应用条右侧 × / 再点一次顶部按钮。 */
  const bulk = { on: false, sel: new Set() };

  const bulkBar = h("div", "h3d-bulkbar");
  bulkBar.style.display = "none";
  const bulkLbl = h("span", "lbl", "");
  bulkBar.appendChild(bulkLbl);
  // 四个分类胶囊：点一个 = 对选中卡片统一应用
  [["角色", "k-char"], ["环境", "k-scene"], ["道具", "k-prop"], ["自行判断", "k-other"]]
    .forEach(([label, cls]) => {
      const pill = h("div", "pill " + cls, label);
      pill.onclick = () => applyBulk(label);
      bulkBar.appendChild(pill);
    });
  const bulkCancel = h("div", "cancel", "×");
  bulkCancel.title = "退出批量模式（ESC）";
  bulkCancel.onclick = () => exitBulk();
  bulkBar.appendChild(bulkCancel);
  bd.appendChild(bulkBar);

  function KIND_OF_LABEL(lbl) {   // 面板文案 → 内部 kind 值
    return lbl === "环境" ? "场景" : lbl === "自行判断" ? "其它" : lbl;
  }

  function syncBulkBar() {
    const n = bulk.sel.size;
    bulkLbl.textContent = n ? "已选 " + n + " 张 →" : "勾选要改的参考图";
  }

  function applyBulk(label) {
    const kindVal = KIND_OF_LABEL(label);
    grid.querySelectorAll(".h3d-ref[data-slot]").forEach((card) => {
      const n = parseInt(card.dataset.slot, 10);
      if (!bulk.sel.has(n)) return;
      card.dataset.mode = kindVal;
      card.dataset.kind = kindVal;
    });
    save();
    exitBulk();
    render();
  }

  function enterBulk() {
    bulk.on = true;
    bulk.sel.clear();
    btnBulk.textContent = "完成";
    btnBulk.classList.add("is-on");
    grid.querySelectorAll(".h3d-ref[data-slot]").forEach((c) => c.classList.add("is-selecting"));
    bulkBar.style.display = "flex";
    syncBulkBar();
  }

  function exitBulk() {
    bulk.on = false;
    bulk.sel.clear();
    btnBulk.textContent = "批量改分类";
    btnBulk.classList.remove("is-on");
    grid.querySelectorAll(".h3d-ref[data-slot]").forEach((c) => {
      c.classList.remove("is-selecting");
      c.classList.remove("is-selected");
    });
    bulkBar.style.display = "none";
  }

  /* ESC 退出多选；只在批量模式下拦截，别抢 ComfyUI 自己的快捷键。
   * ⚠ 加能力判断：离线测试桩的假 DOM 没实现 addEventListener，直接调会 TypeError。
   *   真浏览器里 bd 是正常 div，行为不受影响。 */
  if (typeof bd.addEventListener === "function") {
    bd.addEventListener("keydown", (e) => {
      if (bulk.on && e.key === "Escape") { e.stopPropagation(); exitBulk(); }
    });
  }

  /* 面板里的「从已有图片里挑一张」下拉已按用户要求移除（2026-09-16 第 11 轮）。
   * 移除理由：它和画布上的 LoadImage 是**同一件事的两条路径**，而下拉这条路径
   * 改的是 refOverride/上游控件，容易和画布状态打架，正是「换图没反应」的温床。
   * 现在只剩一条路：画布上的 LoadImage 选图 → 面板自动刷新（见 mountPanels 的
   * 参考图变化统一监听）。要加新图才用面板的「添加图片」上传。 */

  collapsible(node, bd, head.title, head.badge);

  function classify() { return state.classify(); }
  function save() {
    const rows = [];
    const covered = new Set();
    grid.querySelectorAll(".h3d-ref[data-slot]").forEach((card) => {
      const slot = parseInt(card.dataset.slot, 10);
      const kind = card.dataset.kind || "";
      const name = card.dataset.name || "";
      // mode 一起存：不存的话下次 render 会退回 auto，手动指定的分类就白点了
      const mode = card.dataset.mode || "auto";
      covered.add(slot);
      rows.push({ slot, kind, name, mode });
    });
    // ★ 空槽不再渲染（见 render()），但**不能因为它这次没画出来就把以前存的
    // 分类抹掉** —— save() 是从 grid 里扫卡片的，空槽没卡片就会被静默丢弃，
    // 下次接上图分类就丢了。这里把没被覆盖到的历史行补回去。
    classify().forEach((r) => {
      if (r && Number.isFinite(r.slot) && !covered.has(r.slot)) rows.push(r);
    });
    rows.sort((a, b) => (a.slot || 0) - (b.slot || 0));
    const w = findWidget(node, "ref_classify");
    if (w) w.value = JSON.stringify(rows);
    state.onClassify(rows);
    refreshTag();
  }

  /** 把本地图片上传到 ComfyUI 的 input 目录 */
  async function uploadLocal(file) {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    const data = await res.json();
    if (!data || !data.name) throw new Error("上传未返回文件名");
    return data.name;
  }

  /* 找第一个**可用**的参考图槽位（1 起）。
   *
   * 坑 1："输入口存在"不等于"槽位被占用" —— AutoGrow 会预生成一串空口，
   *   光看 input 存在会把 5 个空口全算成已占，然后跑去找第 6 个不存在的口。
   *   空位判定：没接线，或者接了但上游还没选图。
   * 坑 2（★ 本次修的）："输入口不存在"也不等于"槽位满了"。AutoGrow 上限是 9，
   *   但节点上**当前只建了几个口**是另一回事（本工作流存盘时只有 5 个：
   *   ref_image_0..4）。原实现遇到不存在的口就 `continue`，扫完 return 0，
   *   于是第 5 个口接满后面板直接报「9 个参考图槽位都满了，先移除一张再加」
   *   —— 实际上 9 个口连建都没建出来，永远加不到第 6/7 张。
   *   正确做法：已建的口全占满时，返回「下一个待建的口」，交给 refPush 去
   *   addInput 补齐（refPush 本来就会建口，只是从来没人给它这个机会）。 */
  function firstFreeRefSlot() {
    let built = 0;                       // 节点上已经建出来的口数
    for (let i = 0; i < 9; i++) {
      const name = "ref_images.ref_image_" + i;
      const inp = (node.inputs || []).find((x) => x.name === name);
      if (!inp) break;                   // 再往后的口都还没建，不用找了
      built = i + 1;
      let up = null;
      try { up = upstream(node, name); } catch (e) { up = null; }
      const file = up ? String(readWidget(up, "image") || "") : "";
      if (!up || !file) return i + 1;    // 已建但空着 → 优先复用，不新建
    }
    return built < 9 ? built + 1 : 0;    // 全占满 → 开一个新口；真到 9 才是满
  }

  /* 把文件名**真正写回画布上**的参考图节点。
   * 槽位是 1 起的（面板显示 P1/P2…），对应输入 ref_images.ref_image_{slot-1}。
   *   - 上游已经有节点 → 直接改它的 image 控件（LoadImage 的真身）
   *   - 上游是空的   → 现建一个 LoadImage，接到这个输入口上
   *   - 连输入口都没有 → 给节点补一个（AutoGrow 上限 9）
   * 返回 null 表示成功；否则返回一句给用户看的错误说明。
   * （以前只写 state.refOverride，那只是面板内存里的影子，
   *   渲染读的是 LoadImage 的 widget，所以"换了图没反应"。） */
  function refPush(slot, filename) {
    const inputName = "ref_images.ref_image_" + (slot - 1);
    let inp = (node.inputs || []).find((x) => x.name === inputName);

    if (!inp) {
      if (slot > 9) return "参考图最多 9 张";
      if (typeof node.addInput !== "function") {
        return "画布上没有 " + inputName + " 这个输入口";
      }
      try {
        node.addInput(inputName, "IMAGE");
      } catch (e) {
        return "补输入口失败：" + ((e && e.message) || e);
      }
      inp = (node.inputs || []).find((x) => x.name === inputName);
      if (!inp) return "补了输入口但没找到 " + inputName;
    }

    let up = null;
    try { up = upstream(node, inputName); } catch (e) { up = null; }

    if (up) {
      if (setWidget(up, "image", filename)) return null;
      return "上游是 " + (nodeClass(up) || "未知节点") + "，没有 image 控件，改不了";
    }

    const LG = (typeof LiteGraph !== "undefined" && LiteGraph)
      || (typeof window !== "undefined" && window.LiteGraph);
    if (!LG || typeof LG.createNode !== "function") {
      return "拿不到 LiteGraph，建不了 LoadImage 节点";
    }
    let ln = null;
    try { ln = LG.createNode("LoadImage"); } catch (e) { ln = null; }
    if (!ln) return "画布上没有 LoadImage 这个节点类型";

    try {
      app.graph.add(ln);
      ln.pos = [(node.pos ? node.pos[0] : 0) - 300,
                (node.pos ? node.pos[1] : 0) + 40 + (slot - 1) * 90];
      setWidget(ln, "image", filename);
      ln.connect(0, node, (node.inputs || []).indexOf(inp));
    } catch (e) {
      return "接线失败：" + ((e && e.message) || e);
    }
    return null;
  }

  let _refErr = "";

  function assign(slot, filename) {
    _refErr = refPush(slot, filename) || "";
    // 只有"写不回画布"时才退回面板内存里存着（至少缩略图能看），
    // 并在下面那行提示里说清楚 —— 免得用户以为换成功了。
    state.refOverride = state.refOverride || {};
    if (_refErr) state.refOverride[slot] = { file: filename };
    else delete state.refOverride[slot];
    save();
    render();
  }

  btnAdd.onclick = (e) => { e.stopPropagation(); _targetSlot = null; fileInput.click(); };
  fileInput.onchange = async () => {
    const f = fileInput.files && fileInput.files[0];
    fileInput.value = "";
    if (!f) return;
    try {
      const name = await uploadLocal(f);
      if (typeof _targetSlot === "number") {
        assign(_targetSlot, name);
      } else {
        // 复用已存在但空着的输入口，别去找第 6 个不存在的槽位
        const free = firstFreeRefSlot();
        if (!free) {
          _refErr = "9 个参考图槽位都满了，先移除一张再加";
          render();
          return;
        }
        assign(free, name);
      }
    } catch (err) {
      console.error("[H3 Director] 参考图上传失败：", err);
    }
  };

  function refreshTag() {
    // 参考图这边一有变动（换图 / 断线 / 上传），提示词里手写的 <Picture N>
    // 胶囊状态就要跟着变 —— textarea 自己不会发 input 事件，这里是刷新入口。
    refreshTokenViews();
    const used = new Set((state.packData?.segments || [])
      .flatMap((s) => s.pictures || []));
    const have = new Set(refSlotImages(node).filter((s) => s.file).map((s) => s.slot));
    Object.keys(state.refOverride || {}).forEach((k) => have.add(parseInt(k, 10)));
    const missing = [...used].filter((n) => !have.has(n)).sort((a, b) => a - b);

    /* 顶部统计（第 14 轮重设计）：
     * 旧版是「7 参考图 需 1/2/3/4」—— 数量和缺失挤成一坨，分不清哪个是哪个。
     * 现在拆成两个语义明确的东西：
     *   refStat = 「已接 N / 剧本需 M」（主信息，灰底中性）
     *   tag     = 「缺 1/2/3/4」警示条（只在真缺时出现，黄底，不抢主信息） */
    // 没有剧本就没有「需 M」，整块统计隐藏（只留按钮那一行）
    statEl.style.display = used.size ? "" : "none";
    statNum.textContent = String(have.size);
    statSep.textContent = "/";
    statNeed.textContent = String(used.size);

    if (!used.size) { missEl.className = "miss is-ok"; missEl.textContent = ""; return; }
    if (missing.length) {
      missEl.className = "miss";
      missEl.textContent = "缺 " + missing.join("/");
      missEl.title = "剧本引用了这些 <Picture N>，但对应槽位还没接图 —— H3 会自己补出角色";
    } else {
      missEl.className = "miss is-ok";
      missEl.textContent = "已齐";
      missEl.title = "剧本引用的 <Picture N> 全部有对应参考图";
    }
  }

  const KIND_CLS = { "角色": "k-char", "场景": "k-scene",
                     "道具": "k-prop", "其它": "k-other" };
  /* 内部类名 → 面板上用户看到的叫法。"场景"在提示词里是场景，
   * 面板上叫"环境"；"其它"是兜底，用户原话叫"自行判断"。
   * ⚠ 只改徽标文案，KIND_CLS / 存盘的 kind 值一律不动 —— 后者是数据。 */
  const KIND_LABEL = { "角色": "角色", "场景": "环境",
                       "道具": "道具", "其它": "自行判断" };

  /* ---- 参考图分类「自动检测」（2026-09-16 第 11 轮）----
   * 旧实现：要一张张点卡片，循环切 角色→场景→道具→其它。纯手工、容易忘，
   * 而且换图后分类不会自己变，滞后于实际内容。
   *
   * 改成从 PACK 自己的文本里读，三级信号叠加投票，取票数最高的；都没命中就
   * 返回空（不显示分类）—— **宁可不标，也不乱标**：猜错的标签比没有标签更糟，
   * 因为它会进 ref_classify 并被后端 slot_labels 当成显示名用。
   *
   *   信号 2（主力，每次命中 1 票）：扫全部六段字段，取 <Picture N> 前后各 60 字的
   *       窗口，按下面的关键词表投票。窗口而不是全文 —— 全文会把别的图的
   *       修饰词也算进来。
   *   信号 1（仅兜底）：信号 2 一票都没投出来时，才看 <Picture N> 是否出现在
   *       subject_definitions，是就判角色。**不能反过来让信号 1 加权** ——
   *       官方 PACK 会把场景也写进这块（本机 <Picture 3> 就是卧室场景）。
   *   信号 3（兜底）：文件名关键词（有些文件名自带 scene / char 之类）。
   *
   * 手动点卡片仍然可以覆盖，覆盖值连同 mode 一起存进 ref_classify，
   * 后端 slot_labels 只看 kind 字段，不受影响。 */
  const KIND_WORDS = [
    ["角色", ["角色", "人物", "主角", "主人公", "男子", "女子", "男人", "女人",
              "少年", "老人", "女孩", "男孩", "主播", "老师", "学生", "肖像",
              "character", "person", "protagonist", "hero", "heroine", "man",
              "woman", "boy", "girl", "host", "figure", "portrait", "face"]],
    ["场景", ["场景", "背景", "环境", "房间", "卧室", "客厅", "街道", "室外", "室内",
              "城市", "森林", "办公室", "咖啡", "远景", "空镜", "全景",
              "scene", "background", "environment", "room", "bedroom", "street",
              "landscape", "city", "office", "interior", "exterior"]],
    ["道具", ["道具", "物品", "物件", "手机", "杯子", "背包", "书本", "武器",
              "手表", "特写", "产品",
              "prop", "object", "item", "phone", "cup", "bag", "book",
              "weapon", "watch", "product"]],
  ];

  /* 英文词必须按**词边界**匹配，否则 "car" 命中 "camera"、"man" 命中 "woman"。
   * 允许 s / es 复数后缀（characters / props 也能认）；中文没有词边界，走子串。
   * （这个坑和后端 prompt_pack 里 <Picture N> 的正则必须同语义是同一类问题。）
   *
   * 正则**预编译一次**：autoKind 每次 render 要对「9 个槽 × <Picture N> 每处出现 ×
   * 全部关键词」跑一遍，在循环里现造 RegExp 是白白的开销。 */
  function compileMatcher(w) {
    const lw = String(w || "").toLowerCase();
    if (!lw) return null;
    if (/^[a-z0-9]+$/.test(lw)) {
      return new RegExp("(^|[^a-z0-9])" + lw + "(s|es)?([^a-z0-9]|$)");
    }
    return lw;                       // 中文：存字符串，匹配时走 indexOf
  }
  function matchWord(m, hay) {
    if (m == null) return false;
    return typeof m === "string" ? hay.indexOf(m) >= 0 : m.test(hay);
  }
  const KIND_MATCHERS = KIND_WORDS.map(
    (pair) => [pair[0], pair[1].map(compileMatcher)]);

  /** 某个段里六个字段的文本（没有 full_fields 就退回 prompt 正文）。 */
  function segTexts(seg) {
    const ff = seg && seg.full_fields;
    if (ff && typeof ff === "object") {
      return Object.keys(ff).map((k) => String(ff[k] || ""));
    }
    return [String((seg && seg.prompt) || "")];
  }

  function autoKind(slot, file) {
    const votes = new Map();
    const bump = (kind, n) => votes.set(kind, (votes.get(kind) || 0) + n);
    const segs = (state.packData && state.packData.segments) || [];
    if (!segs.length && !file) return "";

    // 信号 2：全部字段里 <Picture N> 前后 60 字窗口投票
    segs.forEach((seg) => {
      segTexts(seg).forEach((t) => {
        if (!t) return;
        const rx = new RegExp("<\\s*Picture\\s*" + slot + "\\s*>", "gi");
        let m;
        while ((m = rx.exec(t))) {
          const win = t.slice(Math.max(0, m.index - 60),
                              m.index + m[0].length + 60).toLowerCase();
          KIND_MATCHERS.forEach((pair) => {
            pair[1].forEach((mm) => { if (matchWord(mm, win)) bump(pair[0], 1); });
          });
        }
      });
    });

    /* 信号 1：官方「主体定义」块 —— **只在上下文完全没词可依时才用**。
     *
     * ★ 这里踩过一个真实的坑：最早的实现是无条件 bump("角色", 3)，结果本机这份
     * 工作流直接翻车 —— <Picture 3> 在 subject_definitions 里写的是
     * "<Picture 3> is the bedroom set reference; the room, its props..."，
     * 它明明是**卧室场景**，却因为"出现在主体定义块"被 +3 压成了角色。
     * 官方 PACK 会把场景也写进这一块，所以**块名不是可靠的类别信号**。
     *
     * 修法：subject_definitions 只在关键词投票**一票都没有**时兜底。
     * 有票就让票说话 —— 票是从 <Picture N> 前后 60 字的真实描述里来的，更准。 */
    if (!votes.size) {
      const rxSub = new RegExp("<\\s*Picture\\s*" + slot + "\\s*>", "i");
      const inSubject = segs.some((seg) => {
        const ff = seg && seg.full_fields;
        const t = ff ? String(ff.subject_definitions || "") : "";
        return !!t && rxSub.test(t);
      });
      if (inSubject) return "角色";
    }

    let best = "", bestN = 0;
    votes.forEach((n, k) => { if (n > bestN) { bestN = n; best = k; } });
    if (best) return best;

    // 信号 3：文件名关键词兜底（剧本没提这张图时，只看名字能看出什么）
    const fn = String(file || "").toLowerCase();
    for (let i = 0; i < KIND_MATCHERS.length; i++) {
      if (KIND_MATCHERS[i][1].some((mm) => matchWord(mm, fn))) return KIND_MATCHERS[i][0];
    }
    return "";
  }

  /** 剧本里每个 <Picture N> 被多少分镜引用过。槽位编号与 N 一一对应。 */
  function pictureUsage() {
    const counts = new Map();
    const needed = new Set();
    (state.packData?.segments || []).forEach((seg) => {
      (seg.pictures || []).forEach((raw) => {
        const n = Number(raw);
        if (!Number.isFinite(n)) return;
        needed.add(n);
        counts.set(n, (counts.get(n) || 0) + 1);
      });
    });
    return { counts, needed };
  }

  function makeCard(slot, savedRow, slotInfo, usage) {
    const ov = state.refOverride && state.refOverride[slot];
    const file = (ov && ov.file) || (slotInfo && slotInfo.file) || "";
    const count = (usage && usage.counts.get(slot)) || 0;
    const needed = !!(usage && usage.needed.has(slot));
    const card = h("div", "h3d-ref");
    card.dataset.slot = String(slot);
    card.dataset.kind = (savedRow && savedRow.kind) || "";
    card.dataset.name = (savedRow && savedRow.name) || "";
    if (!file) card.classList.add("is-empty");
    // ★ 剧本点名了 <Picture N> 但这槽没图：以前是静默的，H3 会自己补人。
    if (!file && needed) card.classList.add("is-missing");

    if (file) {
      const img = document.createElement("img");
      img.src = viewUrl(file, "input");
      img.onerror = () => { img.style.visibility = "hidden"; };
      card.appendChild(img);
    } else {
      card.appendChild(h("div", "ph", "∅"));
    }

    /* 状态条：左侧 4px 色边（::before），替代旧版压在脸上的文字徽标。
     * 未引用（接了图但剧本没点名）= 黄；缺图（剧本点名但没接）= 红。 */
    if (!file && needed) card.classList.add("is-missing");
    else if (file && !count) card.classList.add("is-unused");

    /* ---- 遮罩层 .veil（第 16 轮）：hover 时整卡压半透明灰底，数据+功能居中排列 ----
     * 第 14/15 轮把「引用次数」放右上角标、「操作按钮」放右侧竖排工具条，
     * 结果是四个角都有东西，用户要满卡找。现在全部收进一层遮罩：
     *   ① .vtop  顶部细行：左=引用次数，右=槽位号「图片N」
     *   ② .vkind 中间大字：分类（主信息，最醒目）
     *   ③ .vacts 底部横排：三枚 30px 圆钮
     * 默认 opacity:0，hover 才整体浮现 —— 「单一 hover 层」原则。 */
    const veil = h("div", "veil");

    /* ① 顶部细行：引用次数（左）+ 槽位号（右）。
     * 次数语义：有引用=次数，接了图没被引用=黄「0 镜」，缺图=红「! 缺图」。 */
    const vtop = h("div", "vtop");
    const cnt = h("span", "cnt");
    if (count) {
      cnt.textContent = count + " 镜";
      cnt.title = "被 " + count + " 个分镜引用（<Picture " + slot + ">）";
    } else if (file) {
      cnt.textContent = "0 镜";
      cnt.className = "cnt is-warn";
      cnt.title = "未被引用：剧本里没有任何 <Picture " + slot + "> 指向它，H3 不会把它画进画面";
    } else if (needed) {
      cnt.textContent = "! 缺图";
      cnt.className = "cnt is-bad";
      cnt.title = "缺图：剧本引用了 <Picture " + slot + ">，但这个槽位还没接图";
    } else {
      cnt.textContent = "—";
      cnt.title = "空槽位：既没接图，剧本也没点名";
    }
    vtop.appendChild(cnt);
    const sid = h("span", "sid", "图片" + slot);
    sid.title = "槽位序号 " + slot + " —— 提示词里用 <Picture " + slot + "> 引用它";
    vtop.appendChild(sid);
    veil.appendChild(vtop);

    /* ② 中间分类大字（内容由 paintKind 在 kind 声明之后填） */
    const vkind = h("div", "vkind");
    veil.appendChild(vkind);

    /* ③ 底部操作按钮横排：↻ 替换 / ⇄ 切分类 / × 删除。
     * ▾「从 input 已有图里挑一张」**不恢复** —— 它和画布上的 LoadImage 是同一件
     * 事的两条路径，两条路径互相打架正是「换图没反应」的根源。 */
    const vacts = h("div", "vacts");
    const rep = h("div", "vact", "↻");
    rep.title = "替换这张参考图（上传新文件，直接写回画布上的 LoadImage）";
    rep.onclick = (e) => { e.stopPropagation(); _targetSlot = slot; fileInput.click(); };
    vacts.appendChild(rep);
    const cyc = h("div", "vact is-cycle", "⇄");
    cyc.title = "切换分类（自动 / 角色 / 环境 / 道具 / 自行判断）";
    cyc.onclick = (e) => { e.stopPropagation(); cycleKind(); };
    vacts.appendChild(cyc);
    const del = h("div", "vact is-danger", "×");
    del.title = "移除这张参考图（断开画布上的连线，不删文件）";
    del.onclick = (e) => {
      e.stopPropagation();
      // 光删 refOverride 没用 —— 得把这条连线真的断掉，渲染才不再用它
      const list = node.inputs || [];
      const idx = list.indexOf(
        list.find((x) => x.name === "ref_images.ref_image_" + (slot - 1)));
      if (idx >= 0) {
        try { node.disconnectInput(idx); } catch (err) { /* 本来就没连 */ }
      }
      if (state.refOverride) delete state.refOverride[slot];
      _refErr = "";
      save();
      render();
    };
    vacts.appendChild(del);
    veil.appendChild(vacts);

    card.appendChild(veil);

    /* 分类：默认**自动检测**，手动点过就以手动值为准。
     * mode 存进 ref_classify（"auto" / "角色" / "场景" / "道具" / "其它"），
     * kind 是解析后的显示值 —— 后端 slot_labels 只读 kind，不受影响。 */
    const MODES = ["auto", "角色", "场景", "道具", "其它"];
    /* ★ 老数据兼容：以前只有 kind 没有 mode。如果一律当成 auto，自动检测会把
     * 用户早就手工标好的分类（本机这份工作流里就有
     * {"slot":1,"kind":"角色","name":"王总（S1·男）"}）**静默覆盖掉** —— 那是要命的。
     * 所以：有 mode 用 mode；没有 mode 但 kind 非空 → 视为手工，原样保留；
     * 两者都没有（或值非法）才是 auto。想让某张卡改成自动，点它切回「自动」即可。 */
    let mode = "auto";
    if (savedRow) {
      if (MODES.indexOf(savedRow.mode) >= 0) mode = savedRow.mode;
      else if (MODES.indexOf(savedRow.kind) >= 0) mode = savedRow.kind;
    }
    const kind = mode === "auto" ? autoKind(slot, file) : mode;
    card.dataset.mode = mode;
    card.dataset.kind = kind;

    /* 分类一图两态（第 16 轮）：
     *   · 默认态 → 左下角 9px 色点（唯一的常驻徽标，颜色即分类）
     *   · hover 态 → 圆点隐去，遮罩中间浮出分类大字
     * 两个元素承载同一份数据，所以**必须走同一个 paintKind 刷**，
     * 否则点了切换只更新一个、另一个停在旧分类（第 13 轮踩到过）。
     * 手动覆盖 = 圆点外加白圈，一眼区分"系统判的 / 自己定的"。
     * ★ 必须在 kind/mode 声明之后：const 有 TDZ，放前面会抛
     *   "Cannot access 'kind' before initialization"（第 13 轮踩过）。 */
    let dot = null;

    /* 把 (mode, kind) 同时刷到圆点 + 遮罩分类大字，并触发「刚被点过」色环。
     * 初次渲染和点击切换**必须走同一个函数**。 */
    function paintKind(m, k) {
      // ① 圆点：没分类就整个隐藏（宁可不标，也不冒充一个分类）
      if (dot) {
        if (!k) { dot.style.display = "none"; }
        else {
          dot.style.display = "";
          dot.className = "dot " + (KIND_CLS[k] || "k-other") +
                          (m !== "auto" ? " is-manual" : "");
          dot.title = kindTitle(m, k);
          // 第 15 轮：把环颜色设成分类色，配合 is-just-clicked 触发脉冲
          // ⚠ 能力判断：离线测试桩的假 DOM 没实现 setProperty
          if (typeof dot.style.setProperty === "function") {
            dot.style.setProperty("--h3d-dot-ring",
              ({"k-char":"#4fff8f","k-prop":"#d68cf0","k-scene":"#87b6eb","k-other":"#b0b0b0"}
                [KIND_CLS[k] || "k-other"]) || "#4fff8f");
          }
        }
      }
      // ② 遮罩分类大字：面板文案用 KIND_LABEL（内部"场景"→显示"环境"）
      if (k) {
        vkind.className = "vkind " + (KIND_CLS[k] || "k-other");
        vkind.textContent = KIND_LABEL[k] || k;
        vkind.title = kindTitle(m, k);
      } else {
        vkind.className = "vkind is-none";
        vkind.textContent = "未分类 · 点卡片指定";
        vkind.title = kindTitle(m, k);
      }
    }

    if (file && kind) {
      dot = h("div", "dot");
      card.appendChild(dot);
    }
    paintKind(mode, kind);
    if (!file) {   // 空槽位：圆点本就不该有，遮罩里给个"未接图"提示而不是"未分类"
      vkind.className = "vkind is-none";
      vkind.textContent = needed ? "缺图 · 拖图进来" : "空槽位";
    }

    /* 「刚被点过」反馈：圆点外圈 0.5s 出现一道细色环然后收回。
     * 不靠 hover 永久保持（用户反馈：点完回头看画面，啥痕迹都没）。
     * 通过添加 is-just-clicked 类启动 CSS @keyframes，事件结束自动移除。
     * 这里用 onanimationend 而非固定 setTimeout(550) —— 不同时长自动跟上 CSS。 */
    function flashJustClicked() {
      card.classList.remove("is-just-clicked");   // 重启动画
      void card.offsetWidth;                      // 强制 reflow
      card.classList.add("is-just-clicked");
    }

    function kindTitle(m, k) {
      if (m === "auto") {
        return k
          ? "自动判定为「" + k + "」—— 点卡片可手动改（自动 / 角色 / 场景 / 道具 / 其它）"
          : "自动判定不出来（剧本没提它或关键词没命中）—— 点卡片手动指定";
      }
      return "手动指定为「" + m + "」—— 点卡片继续切换，或切回自动";
    }
    // ⚠ 加能力判断：测试桩的假 DOM 没实现 addEventListener
    if (typeof card.addEventListener === "function") {
      card.addEventListener("animationend", (e) => {
        if (e.animationName === "h3d-dot-pulse") card.classList.remove("is-just-clicked");
      });
    }

    /* caption（底部「图片N」条）已取消（第 15 轮）。
     * 理由：① 编号已挪进遮罩层右上角；② 引用次数也在遮罩顶部；
     * ③ 圆点承担分类—— caption 多写一遍只是占用缩略图底部那一块本该干净留白的区域。 */

    /* 多选勾选圈：只在批量模式显形（.is-selecting 时 CSS 才 display:flex） */
    const pick = h("div", "pick");
    card.appendChild(pick);

    /* 切分类：自动 → 角色 → 场景 → 道具 → 其它 → 自动（第 14 轮抽成函数）。
     * 两个入口共用：点卡片本体、遮罩里的 ⇄ 按钮。
     * 就地更新不重渲染，所以圆点 / 遮罩分类大字 / card.title 都要手动刷 ——
     * 走 paintKind，然后触发「刚被点过」色环脉冲动画。 */
    function cycleKind() {
      const cur = MODES.indexOf(card.dataset.mode || "auto");
      const nxt = MODES[(cur + 1) % MODES.length];
      applyKind(nxt);
    }

    /* 写入分类（批量模式和单张切换共用同一条路，避免两处逻辑发散）。 */
    function applyKind(nxt) {
      const k = nxt === "auto" ? autoKind(slot, file) : nxt;
      card.dataset.mode = nxt;
      card.dataset.kind = k;
      if (!dot && k && file) {   // 之前判不出来没建圆点，现在有了 → 补建
        dot = h("div", "dot");
        card.appendChild(dot);
      }
      paintKind(nxt, k);
      card.title = kindTitle(nxt, k);
      flashJustClicked();          // 第 15 轮：色环反馈
      save();
    }

    card.onclick = () => {
      // 批量模式：点卡片 = 勾选/取消，不再是切分类
      if (bulk.on) {
        const n = parseInt(card.dataset.slot, 10);
        if (bulk.sel.has(n)) bulk.sel.delete(n); else bulk.sel.add(n);
        card.classList.toggle("is-selected", bulk.sel.has(n));
        syncBulkBar();
        return;
      }
      cycleKind();
    };
    card.title = kindTitle(mode, kind);
    return card;
  }

  function makeAdd() {
    const tile = h("div", "h3d-ref-add");
    tile.appendChild(h("div", "plus", "+"));
    tile.appendChild(h("div", null, "添加"));
    tile.onclick = () => { _targetSlot = null; fileInput.click(); };
    return tile;
  }

  function render() {
    grid.innerHTML = "";
    const slots = refSlotImages(node);
    const saved = classify();
    /* ★ 空槽不再占位（2026-09-16 第 11 轮）。
     * 旧逻辑是把所有**已建出来的输入口**都塞进 list，于是没接图的 ref_image_4..8
     * 会渲染成一排「∅」占位卡片（用户截图里 图片5/6/7 那一排）。
     * 现在只画**真的有图**的槽：接了图的 slot + refOverride 里带 file 的兜底。
     * 注意别改成「只画 needed 的」：接了图但剧本没引用（未引用徽标）也是要看到的，
     * 那正是「换图没反应」的诊断信息。
     * 缺图告警不会因此丢失 —— 下面 noteRef 仍然会列出 <Picture N> 没接图。 */
    const list = [];
    const seen = new Set();
    slots.forEach((s) => {
      if (!s || !s.file || seen.has(s.slot)) return;
      seen.add(s.slot);
      list.push(s.slot);
    });
    Object.keys(state.refOverride || {}).forEach((k) => {
      const n = parseInt(k, 10);
      const ov = state.refOverride[n];
      if (!Number.isFinite(n) || !ov || !ov.file || seen.has(n)) return;
      seen.add(n);
      list.push(n);
    });
    list.sort((a, b) => a - b);

    const usage = pictureUsage();
    list.forEach((slot) => {
      const info = slots.find((s) => s.slot === slot) || null;
      const row = saved.find((r) => r.slot === slot) || {};
      grid.appendChild(makeCard(slot, row, info, usage));
    });
    grid.appendChild(makeAdd());
    refreshTag();
    // 把"到底有没有接进画布"说清楚 —— 这一步以前是静默的
    if (_refErr) {
      noteRef.textContent = "⚠ " + _refErr + " —— 图已上传，但没接进画布，渲染不会用它";
      noteRef.className = "h3d-note is-warn";
    } else {
      const slotsNow = refSlotImages(node);
      const n = slotsNow.filter((s) => s.file).length;
      const usage = pictureUsage();
      // 双向对账，和后端 ref_images.reconcile 说同一件事：
      // 剧本点名了却没图 → H3 自己补人；接了图剧本没提 → H3 不会画它。
      const missing = [...usage.needed].filter(
        (s) => !slotsNow.some((x) => x.slot === s && x.file)).sort((a, b) => a - b);
      const unused = slotsNow.filter((s) => s.file && !usage.needed.has(s.slot))
        .map((s) => s.slot);
      const notes = [];
      if (n) notes.push("已接入画布 " + n + " 张 · 换图直接写回上游 LoadImage");
      if (missing.length) {
        notes.push("⚠ 剧本引用了 <Picture " + missing.join("/")
          + ">，对应槽位没接图 —— H3 会自己补出角色");
      }
      if (unused.length) {
        notes.push("⚠ 槽 " + unused.join("/")
          + " 接了图但剧本没引用 —— H3 不会把它画进画面");
      }
      if (!notes.length) {
        notes.push("还没接参考图。「添加图片」= 上传新文件；"
          + "要换已有的图，改画布上 LoadImage 的 image 控件，面板会自动跟上。");
      }
      noteRef.textContent = notes.join("　");
      noteRef.className = "h3d-note"
        + ((missing.length || unused.length) ? " is-warn" : "");
    }
    refit(node);   // 参考图增删会改变行数，节点高度跟着贴合
  }

  /* 参考图变化监听**不在这里**做了 —— 提到 mountPanels() 统一调度。
   * 原因：分镜时间线的缩略图也吃 refSlotImages()，只在参考图面板里监听的话，
   * 会出现「参考图卡片换了、时间线缩略图还是旧的」这种半新半旧。
   * 见 mountPanels 里的「参考图变化统一监听」。 */

  sec.appendChild(hd);
  sec.appendChild(bd);
  return { sec, badge: head.badge, render };
}

/* ------------------------------------------------------------------ */
/* 版块 4：分镜时间线 —— 实线已渲/虚线未渲，小马单向奔过并留下蹄印     */
/* ------------------------------------------------------------------ */
function buildTimelinePanel(node, state) {
  const sec = h("div", "h3d-sec");
  const head = sectionHead(0, "分镜时间线");
  const hd = head.hd, tag = head.tag;

  // 渲染模式按钮组：放刷新状态左侧，按钮间 10px 间隔（用户要求）。
  // 「全部渲染」= resume=false，忽略缓存整条重渲；「断点渲染」= resume=true，
  // 从上次 manifest 继续。点完按钮 2s 后自动复位 resume=true（安全的默认），
  // 避免一次点击把后续整条渲染也强制改成不缓存。
  const renderBtns = h("span");
  renderBtns.style.cssText = "display:inline-flex;align-items:center;gap:10px;";
  const btnFull = h("button", "h3d-btn", "全部渲染");
  const btnResume = h("button", "h3d-btn", "断点渲染");
  // 「重渲已改段」：把编辑过提示词的段（及因链式缓存键失效而必须一起重渲的
  // 后续段）从最早那段起重渲，未改的前段复用磁盘缓存。
  const btnRerunDirty = h("button", "h3d-btn", "重渲已改段");
  btnRerunDirty.title = "编辑过分镜提示词的段：从最早改过的那段起重渲到结尾，"
    + "之前的段复用磁盘缓存（提示词改动只影响本段及其后，因为缓存键是链式的）";
  renderBtns.appendChild(btnFull);
  renderBtns.appendChild(btnResume);
  renderBtns.appendChild(btnRerunDirty);
  hd.appendChild(renderBtns);

  const btnRefresh = h("button", "h3d-btn", "刷新状态");
  hd.appendChild(btnRefresh);
  const bd = h("div", "h3d-bd");

  // 进度条
  const barRow = h("div", "h3d-bar-row");
  const bar = h("div", "h3d-bar");
  const barFill = h("i");
  bar.appendChild(barFill);
  const barText = h("span", "h3d-bar-text", "0 / 0");
  barRow.appendChild(bar);
  barRow.appendChild(barText);
  bd.appendChild(barRow);

  // 段块
  const segsRow = h("div", "h3d-segs");
  bd.appendChild(segsRow);

  /* ★ 第 39 轮：run_segments 脏串兜底标志 —— syncRunSegmentsToUI 首次发现
     raw 是脏的（parseRunSegments 解出空集 + raw 非空）就写回 "" 一次，
     之后不再触发（避免 setWidget → dirty → 渲染 → 再次 sync 触发循环）。 */
  let _h3RunSegSanitized = false;

  /* ★ 第 35 轮：run_segments → 段块勾选框视觉同步 —— widget 真值主导。
     任何入口（段块 chk click、工具栏"重渲已改段"、后端回推）改了
     run_segments 后，调用本函数立即把 segsRow 里所有 .run 的 checked
     同步到 widget 真值，不等 panel 全量重绘。
     之前只有 chk.onclick 里手动同步 —— 工具栏按钮改了 widget 之后，
     段块视觉没动，用户以为"勾选不了"或"联动坏了"。 */
  function syncRunSegmentsToUI() {
    // run_segments 是**集合语义**（"1,3,5-7"）—— 勾哪些段号就高亮哪些段块。
    // 后端会按 min(run_set) 自动决定从哪段开始重渲（首尾锚定耦合不可改）。
    // 这里只忠实反映"用户勾了哪些段"，不替用户决定起点。
    const cells = segsRow.querySelectorAll(".h3d-seg");
    const raw = readStr(node, "run_segments", "");
    const set = parseRunSegments(raw, cells.length);
    // ★ 第 39 轮：widget 脏串兜底 —— 浏览器端偶有「0s2s4s6s…」这种垃圾值
    //   留在 run_segments 输入框里（不是 JSON/磁盘问题，是 ComfyUI 运行时状态），
    //   parseRunSegments 已隐式过滤（parseInt("0s…")=0 被 n>=1 守卫挡掉），
    //   但**显示**还是那条脏字符串。等用户**首次**点击任何 chk 时 chk handler
    //   就会写回干净值；下面这个一次性 sanitize 是为了「用户没点也能立刻干净」，
    //   只在 syncRunSegmentsToUI **首次**调用时触发，且只在 raw != "" 且解析为空集
    //   且未 sanitize 过时才写回 ""（避免循环 + 避免误清用户真要清空的「全空」状态）。
    if (!_h3RunSegSanitized && raw && !set.size) {
      _h3RunSegSanitized = true;
      try { setWidget(node, "run_segments", ""); } catch (e) {}
    }
    cells.forEach((c) => {
      const i = Number(c.dataset.index);
      const on = set.has(i);                // ★ 第 38 轮：集合语义，按段号查表
      const cb = c.querySelector(".run");
      if (cb) {
        cb.setAttribute("aria-checked", on ? "true" : "false");
        cb.dataset.run = on ? "on" : "";     // ★ 重新启用：on=勾选、空=未选
      }
      c.classList.toggle("is-run", on);
      // 不再使用 is-run-start —— 集合语义下没有"起点"概念，
      // 起点由后端按 min(run_set) 在执行时确定。
    });
  }

  /* ── 借鉴融合：键盘 + 滚轮导航（参考 AIMixer 时间线的 keydown / wheel）──
     segsRow 加 tabIndex 才能聚焦；键盘只挂在它身上且要求焦点确实在它身上，
     绝不做全局监听 —— 否则会跟 ComfyUI 自身快捷键、提示词 @ 菜单的 keydown
     抢事件。段块 mousedown 时把焦点交回 segsRow，点完即可连续用键盘操作。 */
  segsRow.tabIndex = 0;
  function segCells() {
    return Array.prototype.slice.call(segsRow.querySelectorAll(".h3d-seg"));
  }
  function selectSeg(idx) {
    const cells = segCells();
    if (!cells.length) { return; }
    if (idx < 1) { idx = 1; }
    if (idx > cells.length) { idx = cells.length; }
    // 走既有真相源 state.planSelect（时间线与「功能规划」面板共用同一份），
    // 缺失时才降级为纯视觉选中 —— 绝不绕过真相源自行加/删 class。
    if (state.planSelect) { state.planSelect(idx); }
    else {
      cells.forEach(function (c) {
        c.classList.toggle("is-plan-selected", Number(c.dataset.index) === idx);
      });
    }
    activeIndex = idx;
    const cur = cells[idx - 1];
    if (cur && cur.scrollIntoView) {
      cur.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }
  segsRow.addEventListener("keydown", function (e) {
    if (document.activeElement !== segsRow) { return; }
    const cells = segCells();
    if (!cells.length) { return; }
    const cur = activeIndex > 0 ? activeIndex : 1;
    if (e.key === "ArrowRight") { selectSeg(cur + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { selectSeg(cur - 1); e.preventDefault(); }
    else if (e.key === "Home") { selectSeg(1); e.preventDefault(); }
    else if (e.key === "End") { selectSeg(cells.length); e.preventDefault(); }
    else if (e.key === "Escape") { activeIndex = -1; e.preventDefault(); }
    else if (e.key === "r" || e.key === "R") { repairSegment(cur); e.preventDefault(); }
  });
  // 滚轮横向滚动：只有段数多到溢出时才接管；不溢出就不拦，纵向滚动照常给页面
  segsRow.addEventListener("wheel", function (e) {
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) { return; }
    if (segsRow.scrollWidth <= segsRow.clientWidth + 1) { return; }
    segsRow.scrollLeft += e.deltaY;
    e.preventDefault();
  }, { passive: false });

  /* 第 26 轮起：段块右键菜单整体移除。showSegMenu 函数不再保留——
     cell.oncontextmenu 只 preventDefault 屏蔽浏览器原生菜单，不再挂浮层。
     段块交互入口收敛到三个：单击选中 / 双击全屏 / hover-.re 重渲。 */

  // 时间标尺（A2 → 第 40 轮重写）：段块上方一行秒刻度。只展示、不参与渲染。
  // ★ 与 .h3d-segs 同构：同样的 flex + gap + 每格同样的 flexGrow（= 该段
  //   真实时长），所以尺子上的每一格都压在它对应的分镜缩略图正上方。
  //   刻度间隔由 rulerStep() 决定，保证带数字的刻度不会挤在一起。
  const ruler = h("div", "h3d-ruler");
  bd.insertBefore(ruler, segsRow);

  // 轨道 + 蹄印层 + 小马（蹄印在马下面，免得盖住马腿）
  const rail = h("div", "h3d-rail");
  const railLine = h("div", "h3d-rail-line");
  rail.appendChild(railLine);
  const hoofs = h("div", "h3d-hoofs");
  rail.appendChild(hoofs);
  const pony = ponySvg();
  rail.appendChild(pony.wrap);
  bd.appendChild(rail);

  const status = h("div", "h3d-status", "空闲");
  bd.appendChild(status);

  // 让「在跟踪哪个会话」可见 —— 会话名错了时间线就是空的，而这以前
  // 是静默的（面板猜 my_chain、后端落盘却是 PACK 的 Project）。
  const noteSess = h("div", "h3d-note", "");
  bd.appendChild(noteSess);

  collapsible(node, bd, head.title, head.badge);

  let activeIndex = -1;
  let doneSet = new Set();
  let railSpans = [];          // 每段的轨道块（小马按段跑 + 填充都要用）

  /* ---- 小马：只朝一个方向跑，并且**只在当前这一段的轨道区间里跑** ---- */
  /* 以前是 ping-pong（撞到边界就掉头），看起来像"来回踱步"；后来改成一律
   * 向右、跑出右端回左端。现在是更准的一条：正在渲第 N 段，小马就只在第 N
   * 段那截轨道上来回跑，跑到段尾回到段首 —— 一眼能看出"在渲哪一段、渲到
   * 这一段的哪儿"，而不是满轨道乱窜。没在渲（准备 / 空闲）时才跑全程。 */
  const PONY_W = 32;
  const PONY_SPEED = 1.5;      // px / 帧（≈90 px/秒）
  const HOOF_STEP = 14;        // 每前进 14px 落一只蹄印
  const HOOF_MAX = 60;         // 蹄印数量上限（animationend 没触发时的兜底）
  let _raf = null, _x = 0, _running = false, _hoofAcc = 0, _hoofAlt = false;
  let _t0 = 0;                 // 起跑时刻，用来算步态相位

  function span() { return Math.max(0, (rail.clientWidth || 0) - PONY_W); }

  /** 当前正在渲的那一截轨道的 [起点, 终点]（小马左边界的可跑范围）。
   *  没在渲任何一段时返回 null —— 那就跑全程。 */
  function activeRange() {
    const el = activeIndex >= 1 ? railSpans[activeIndex - 1] : null;
    if (!el || !el.offsetParent) return null;
    const total = rail.clientWidth || 0;
    const w = el.offsetWidth || 0;
    if (!total || w <= 0) return null;
    const a = Math.max(0, Math.min(el.offsetLeft || 0, total - PONY_W));
    const b = Math.max(a, Math.min(total - PONY_W, a + w - PONY_W));
    return (b > a) ? [a, b] : null;
  }

  /** 小马脚下的线：p=0 全是暗虚线，p=1 整段是亮实线。
   *  由虚到实 —— .fill 盖在 .base 上，宽从 0 长到 100%；
   *  由暗到亮 —— 颜色从 #1e5a38 插值到 #4fff8f，辉光同步变强。 */
  const RAIL_DIM = [30, 90, 56];
  const RAIL_LIT = [79, 255, 143];

  function setFill(fill, p) {
    if (!fill) return;
    p = Math.min(1, Math.max(0, Number(p) || 0));
    fill.style.width = (p * 100).toFixed(1) + "%";
    const c = RAIL_DIM.map((v, i) =>
      Math.round(v + (RAIL_LIT[i] - v) * p));
    fill.style.background = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
    fill.style.boxShadow = p > 0.02
      ? "0 0 " + (3 + 5 * p).toFixed(1) + "px rgba(79,255,143,"
        + (0.12 + 0.5 * p).toFixed(2) + ")"
      : "none";
  }

  /** 按 doneSet / activeIndex 重画整条轨道。activeProgress 只给当前段。
   *
   *  ★ 顺序很重要：**正在渲的那一段优先**，哪怕它以前渲过（doneSet 里有）。
   *    否则重渲已渲染的段时，轨道会直接画成 100% 亮实线，完全不跟小马走 ——
   *    "由虚到实 / 由暗到亮"就看不见了（真浏览器抓到过：fill 恒为 100%）。
   *    正在渲 = 这一段的进度由小马位置决定，渲完才归到 doneSet 那条分支。 */
  function paintRail(activeProgress) {
    railSpans.forEach((el, i) => {
      if (!el) return;
      const idx = i + 1;
      const fill = el.querySelector ? el.querySelector(".fill") : null;
      if (!fill) return;
      if (idx === activeIndex) setFill(fill, activeProgress || 0);
      else if (doneSet.has(idx)) setFill(fill, 1);
      else setFill(fill, 0);
    });
  }

  function clearHoofs() {
    hoofs.innerHTML = "";
    _hoofAcc = 0;
  }

  function dropHoof(x) {
    _hoofAlt = !_hoofAlt;
    const el = h("i", "h3d-hoof");
    el.style.left = Math.round(x + 2) + "px";
    // 左右蹄交错：一只落在线上，一只稍低，读起来才像四蹄交替
    el.style.bottom = (_hoofAlt ? 7 : 9) + "px";
    el.addEventListener("animationend", () => {
      try { el.remove(); } catch (e) { /* 已经没了 */ }
    });
    hoofs.appendChild(el);
    // 超上限就摘掉最老的一只（remove() 已经从父节点摘走，别再 removeChild）
    if (hoofs.children.length > HOOF_MAX) {
      const old = hoofs.children[0];
      if (old) {
        try {
          if (typeof old.remove === "function") old.remove();
          else hoofs.removeChild(old);
        } catch (e) { /* 已经被 animationend 摘掉了 */ }
      }
    }
  }

  /* 步态时钟。rAF 回调的 ts 参数在不同环境（含离线 mock）不一定有，
   * 用 performance.now() 更稳，再退一步 Date.now()。 */
  function gaitClock() {
    try {
      if (typeof performance !== "undefined" && performance.now) return performance.now();
    } catch (e) { /* 没有就没有 */ }
    return Date.now();
  }

  function step() {
    // 正在渲第 N 段 → 只在这一段里跑；否则（准备 / 未知）跑全程
    const r = activeRange();
    const from = r ? r[0] : 0;
    const to = r ? r[1] : span();
    if (to > from) {
      // 段切换 / 段块重排后 _x 可能落在区间外，先拉回段首
      if (_x < from || _x > to) _x = from;
      _x += PONY_SPEED;
      if (_x >= to) {                // 跑到段尾 → 回到段首再跑一圈
        _x = from;
        clearHoofs();
      }
      pony.wrap.style.left = Math.round(_x) + "px";
      _hoofAcc += PONY_SPEED;
      if (_hoofAcc >= HOOF_STEP) { _hoofAcc = 0; dropHoof(_x); }
      // 脚下的线跟着小马走：跑到哪儿，实线就长到哪儿、亮到哪儿
      if (r) paintRail((_x - from) / (to - from));
    } else {
      pony.wrap.style.left = "0px";
    }
    ponyGait(pony, gaitClock() - _t0);   // 四蹄交替 + 身子起伏
    if (_running) _raf = requestAnimationFrame(step);
  }

  function setPony(running, text) {
    _running = !!running;
    if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    if (_running) {
      _t0 = gaitClock();
      _raf = requestAnimationFrame(step);
    } else {
      // 空闲时停在「已渲染前沿」，蹄印收干净，四蹄回到静止张角
      const segs = state.segments();
      const frac = segs.length ? (doneSet.size / segs.length) : 0;
      _x = Math.round(span() * frac);
      clearHoofs();
      pony.wrap.style.left = _x + "px";
      ponyGait(pony, 0);
      paintRail(1);   // 停下来了：当前段（如果有）算跑满，其余按 done 画
    }
    if (text) {
      // 解析成 caption / shot_id 徽标 / task 绿色徽标；纯文字（空闲/错误/断点提示）走兜底。
      _renderStatusText(status, text, running);
    }
    node.setDirtyCanvas(true, true);
  }

  /** 会话名 —— 决定时间线去查哪个会话目录，**必须和后端落盘用的同名**。
   *
   *  解析顺序（**与解析动作解耦**，见下）：
   *    1. 本节点自己的 session_name 控件（老工作流还能手动覆盖）
   *    2. 画布上 H3PromptPackParser 的 session_name 控件 —— 这就是后端
   *       H3PromptPackParser.execute / H3Director 落盘时用的**同一份来源**
   *    3. PACK 解析下发的 session_name（点过「解析 PACK」才有）
   *    4. my_chain（兜底）
   *
   *  ★ 为什么必须补第 2 步：H3Director 已经把 session_name 控件删了（冗余 widget
   *    清理），而 state.packData 只有点过「解析 PACK」才会被填充。于是"先渲染、
   *    隔天再点断点渲染"这条最正常的路径下，第 3 步拿不到值 → 面板退回
   *    my_chain → 去查一个不存在的会话目录 → done=0 → 表面看就是"断点渲染
   *    又从第 1 段开始了"。直接从画布 parser 控件读，和是否点过解析无关。
   */
  function sessionName() {
    const w = findWidget(node, "session_name");
    const v = w && String(w.value || "").trim();
    if (v) return v;
    // 第 2 步：与后端同源 —— 画布上 PACK 解析器的会话名控件
    try {
      const pkNode = findNodeByClass("H3PromptPackParser");
      const pkSess = readStr(pkNode, "session_name", "");
      if (pkSess) return pkSess;
    } catch (e) { /* parser 不在画布上：继续往下退 */ }
    const pk = state.packData;
    const fromPack = pk && String(pk.session_name || "").trim();
    return fromPack || "my_chain";
  }

  /** 新窗口/新标签打开这一段的完整视频（ComfyUI 的 /view 直链）。 */
  function openFullVideo(url) {
    try {
      const a = document.createElement("a");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      a.remove();
      return;
    } catch (e) { /* 走不通就退回 window.open */ }
    try { window.open(url, "_blank", "noopener"); } catch (e) { /* 被拦就算了 */ }
  }

  /** 本段缩略图：优先用本段 pictures 槽位挂的参考图。
   *
   *  ★ 兜底链是必须的：解析出来的 pictures 可能是空的 —— 典型两种情况：
   *    (a) PACK 里那一段本来就没写 <Picture N>（例如英文版 S01 段落为空，
   *        解析出来 pictures=[] 且 prompt 也空）；
   *    (b) 只写了解析不出的槽位。
   *    没有兜底时时间线那一格就画 .blank 空白，只有同节点的「功能规划」栏
   *    看得见图（那边有兜底）—— 用户看到的就是"时间线缩略图不见了"。
   *
   *  ★★ 第 19 轮修正（用户："分镜缩略图在 pack 后可以根据剧情分镜匹配不同的
   *     角色 环境 道具图，没必要全部都一样"）—— 两个真 bug：
   *
   *   bug 1：抠 <Picture N> 时只扫了 `seg.prompt`。而 routes.py 把 seg.prompt
   *          取成 `fields["detailed_description"]`，**角色定义块
   *          （subject_definitions）不在里面** —— 偏偏 <Picture 1/2/4> 这些
   *          角色引用全写在那个块里。于是永远抠不到角色图，
   *          一路掉到下面的兜底 → 每段都拿同一张。
   *          实测 manifest：detailed_description 只有 <Picture 3>（场景），
   *          subject_definitions 才有 <Picture 1,2,3,4>。
   *          → 改成扫**本段全部字段**（full_fields + full_fields_zh + prompt），
   *            口径与后端 `_picture_slots()` 对齐（它也是扫全部字段）。
   *
   *   bug 2：兜底"本节点第一张已接参考图"是**与段落无关的常量** ——
   *          只要本段抠不到，就必然和别段同图，正是用户看到的"全部都一样"。
   *          → 删掉这个兜底。抠不到就返回空，让调用方画 .blank。
   *            "显示空白"是诚实的信息（这段没绑图），"显示错的图"是撒谎。
   *
   *  ★ 还有一个必须避开的坑（A17）：如果直接扫描文本里出现的**第一个**
   *    <Picture N>，角色图（角色定义块里第一个就是 <Subject 1> 的
   *    <Picture 1>）会永远胜出 —— 用户要的是"角色/环境/道具按剧情匹配"，
   *    而角色定义在每一段都一样，于是又变回"每段同一张"。
   *    → 排序规则：**先场景/道具（能体现这段"在哪儿、干什么"），
   *      再角色**。角色图只在本段确实没有环境/道具引用时才当缩略图。 */
  function refThumbFor(seg) {
    if (!seg) return "";
    const slots = refSlotImages(node);
    const pick = (n) => slots.find((s) => s.slot === n && s.file);
    // 作者填的分类元数据（{slot,kind,name}）—— 最权威，决定谁是"人"谁"非人"
    const metas = segRefMetas();
    // ★★ 第 19 轮修正 2（关键）：**逐段字段**里真正引用到的槽位优先。
    //   （剔掉了每段都一样的 subject_definitions 共享块）
    const perShot = perShotPicSlots(seg);

    // ① 后端已算好的 pictures（就是 _picture_slots 的产出：全字段口径）
    //    —— 但用 perShot 排序，让"逐段引用到的"排最前
    const pics = (seg && seg.pictures) || [];
    for (const p of orderPicsForThumb(seg, pics, metas, perShot)) {
      const hit = pick(p);
      if (hit) return viewUrl(hit.file, "input");
    }

    // ② 兜底 A：本段逐段字段抠到的 <Picture N>（最贴剧情，优先用）
    for (const n of orderPicsForThumb(seg, perShot, metas, perShot)) {
      const hit = pick(n);
      if (hit) return viewUrl(hit.file, "input");
    }

    // ③ 兜底 B：连逐段字段都没有引用 → 才动用全字段（含共享角色块）
    const tags = segAllText(seg).match(/<Picture\s+(\d+)\s*>/gi) || [];
    const nums = [];
    for (const tag of tags) {
      const n = Number(String(tag).replace(/\D+/g, ""));
      if (n && nums.indexOf(n) < 0) nums.push(n);
    }
    for (const n of orderPicsForThumb(seg, nums, metas, perShot)) {
      const hit = pick(n);
      if (hit) return viewUrl(hit.file, "input");
    }

    // ④ 抠不到就留空（★ 不再退到"第一张已接图" —— 那会让所有段同图）
    return "";
  }

  /** 参考图槽位的分类元数据（作者在参考图面板里填的 kind / name）。
   *  可能在 state 里（点过面板后缓存）、也可能只在 ref_classify 控件里。 */
  function segRefMetas() {
    try {
      if (state.classify) {
        const rows = state.classify();
        if (rows && rows.length) return rows;
      }
    } catch (e) { /* state 未就绪：继续读控件 */ }
    try {
      const w = findWidget(node, "ref_classify");
      if (w && w.value) return JSON.parse(w.value) || [];
    } catch (e) { /* 解析失败就当没有元数据 */ }
    return [];
  }

  /* 时间标尺重绘（A2 → 第 40 轮重写）。
     ★ 旧实现的坑：把 0…总时长 的所有 major 刻度**绝对定位**成一整行 <i>，
       而 .h3d-ruler 当时没有一条 CSS（缺 position:relative / height /
       overflow），于是 "0s 2s 4s 6s…40s" 全部叠在一起 —— 用户报的"乱码"
       就是这么来的（数字本身没错，是容器把它们压到了同一处）。
     ★ 新实现：标尺与 segsRow **同构** —— 一样的 display:flex / gap:3px，
       每格一样的 flexGrow（= segTotalSec(seg)，与下面段块**同一个口径**），
       所以每格严丝合缝压在它对应的分镜缩略图正上方。刻度用**百分比**画在
       格子内部：段数变多、节点拉宽拉窄都不会再出现标签互相压字。
     ★ 第 41 轮：格子里**不再写段号**（段块左上角 .no 已有），"对应缩略图"
       改由段分界竖线承载 —— 分界线正对缩略图之间那 3px 的缝。
     宽度随节点宽度变，所以挂到 _railRO（rail 尺寸变化）与每次 render()。 */
  function paintRuler() {
    if (!ruler) return;
    if (bd.classList.contains("collapsed")) return;   // 折叠时不量、不画
    ruler.replaceChildren();
    const segs = state.segments();
    if (!segs.length) return;
    const W = ruler.clientWidth || segsRow.clientWidth || 0;
    if (!W) { requestAnimationFrame(paintRuler); return; }  // 布局未就绪，下一帧再试
    // ★ 与段块宽度同一个口径、同一个下限：new_seconds + handoff_seconds
    //   （段真实时长），且和 render() 里的 grow 逐字一致，否则尺子和缩略图
    //   会错开几像素。
    const lens = segs.map((s) => Math.max(0.55, segTotalSec(s)));
    const total = lens.reduce((a, b) => a + b, 0);
    if (total <= 0) return;
    const pxPerSec = W / total;
    // 主刻度间隔：rulerStep() 保证相邻两个**带数字**的刻度至少隔 46px，
    // 所以标签永远不会互相压字（这正是旧版"0s2s4s…"乱码的解药）。
    const step = rulerStep(total, W);
    // 细刻度 = 主刻度的 1/5，只画竖线不写字；46/5 ≈ 9px 一根，再挤就不画
    const minor = step / 5;
    const drawMinor = minor * pxPerSec >= 4;
    let acc = 0;
    // ★ 第 41 轮：改遍历 lens 而不是 segs —— 段号取消后本循环不再需要 seg
    //   对象本身（段宽早就折进 lens 了）。
    lens.forEach((len, i) => {
      const start = acc;
      acc += len;
      const cellW = len * pxPerSec;   // 估算像素宽，用来决定这一格放得下几个字
      const pct = (t) => (((t - start) / len) * 100).toFixed(3) + "%";
      const rc = h("div", "h3d-rul-cell" + (cellW < 20 ? " is-tiny" : ""));
      rc.style.flexGrow = String(len);
      rc.dataset.index = String(i + 1);
      // ★ 第 41 轮：段号取消 —— 段块左上角 .no 已经有 S01/S02…，标尺上不再
      //   重复一份。标尺靠下面的段分界竖线对齐缩略图，不靠文字。
      // 该段真实时长（右上角小字，与段块右下 .dur 的"新增秒数"互补）
      if (cellW >= 42) rc.appendChild(h("span", "h3d-rul-len", fmtRuler(len)));
      // 段分界：只画一根竖线，**不写数字** ——
      // 段界秒数（11.0 / 23.6 / 36.2…）不是整数，写出来既不好读又容易
      // 和下面的整数主刻度（10s / 20s / 30s）贴在一起变成新"乱码"。
      // 它唯一的任务就是"对齐下面的缩略图分界"。
      rc.appendChild(h("i", "h3d-rul-t0"));
      if (drawMinor && cellW >= 14) {
        for (let t = Math.ceil(start / minor) * minor; t < start + len - 1e-6;
             t += minor) {
          if (Math.abs(t - start) < 1e-6) continue;   // 段界已经有竖线了
          const m = h("i", "h3d-rul-min");
          m.style.left = pct(t);
          rc.appendChild(m);
        }
      }
      // 整数秒主刻度（带数字）：落在本格区间内的 step 整数倍
      for (let t = Math.ceil(start / step - 1e-9) * step;
           t < start + len - 1e-6; t += step) {
        const m = h("i", "h3d-rul-maj");
        m.style.left = pct(t);
        m.appendChild(h("span", null, fmtRuler(t)));
        rc.appendChild(m);
      }
      ruler.appendChild(rc);
    });
    // 收尾：最右端终点线 + 总时长
    const end = h("div", "h3d-rul-end");
    end.appendChild(h("i"));
    end.appendChild(h("span", null, fmtRuler(total)));
    ruler.appendChild(end);
  }

  function render() {
    segsRow.innerHTML = "";
    railLine.innerHTML = "";
    railSpans = [];
    // 浮层挂在 body 上，重建段块时得顺手把上一批清掉，否则会越积越多
    document.querySelectorAll(".h3d-pop.is-body").forEach((el) => el.remove());
    const segs = state.segments();
    // 不再渲染会话名标签 —— 跟下面的「已渲 X / Y」进度条、§9 功能规划头部
    // 重复，且 session name 错了也只是时间线空、不会再误导操作。
    noteSess.textContent = "";
    noteSess.style.display = "none";

    if (!segs.length) {
      segsRow.appendChild(h("div", "h3d-empty",
        "还没有分镜 —— 先点「解析 PACK」"));
      barFill.style.width = "0%";
      barText.textContent = "0 / 0";
      tag.textContent = "";
      setPony(false);
      return;
    }

    // A5：从 run_segments 控件读当前选择集（集合语义："1,3,5-7"），
    //   用于段块勾选框的初始态。后端按 min(run_set) 自决定起点（首尾锚定耦合），
    //   界面只如实显示"勾了哪几段"。
    const runSet = parseRunSegments(readStr(node, "run_segments", ""), segs.length);

    segs.forEach((seg, i) => {
      const idx = i + 1;
      const done = doneSet.has(idx);
      const active = activeIndex === idx;
      const cell = h("div", "h3d-seg "
        + (active ? "is-active" : done ? "is-done" : ""));
      cell.dataset.index = String(idx);
      // 段宽按时长成比例 —— 时间线要反映时间。
      // ★ 第 40 轮：口径从"只算新增"改成"新增 + 锚定尾巴"（segTotalSec），
      //   也就是这一段落盘视频的真实长度（帧数 / fps 就是这个值）。
      //   标尺 paintRuler() 用同一个函数算 flexGrow，尺子才能压在缩略图正上方。
      const grow = Math.max(0.55, segTotalSec(seg));
      cell.style.flexGrow = String(grow);
      // ★ 第 27 轮：段块 tooltip 完全清空。
      //   之前这里写的是「段号 · 新增 Xs · 生成 Ys · 任务类型」——
      //   用户要求 hover 段块时**什么 tooltip 都不要弹**。
      //   信息不丢：段号在左上角 .no、时长在右下 .dur、状态在 .st 绿勾
      //   与 .dot 圆点 —— 全都在画面上，不依赖 tooltip。
      cell.removeAttribute("title");

      // 已渲染 → 用真实视频替换静态缩略图，并且在时间线上就循环播放
      const url = state.renderedUrl(idx);
      if (url) {
        const v = document.createElement("video");
        v.src = url;
        v.muted = true;
        v.loop = true;
        v.autoplay = true;
        v.playsInline = true;
        v.preload = "metadata";
        cell.appendChild(v);
        // 某些环境里 autoplay 属性不生效（省电模式 / 未交互），补一脚
        try { const pr = v.play(); if (pr && pr.catch) pr.catch(() => {}); }
        catch (e) { /* 播不动就当静态封面 */ }
      } else {
        const src = refThumbFor(seg);
        if (src) {
          const img = document.createElement("img");
          img.src = src;
          cell.appendChild(img);
        } else {
          cell.appendChild(h("div", "blank"));
        }
      }

      cell.appendChild(h("div", "no", seg.id));
      const st = h("div", "st "
        + (active ? "is-active" : done ? "is-done" : "is-todo"),
        active ? "●" : done ? "✓" : "○");
      cell.appendChild(st);
      // ★ 第 23 轮新增：参考图同款圆形渲染状态点（与 .st 字符绿勾并存）。
      //   .st 是右上角徽标，.dot 是左下角圆点——同一信息两种视觉语言，
      //   强化参考图与时间线两个区域的视觉一致性。
      const dot = h("div", "dot "
        + (active ? "is-active" : done ? "is-done" : "is-todo"));
      cell.appendChild(dot);
      // 「提示词已改、待重渲」角标：编辑分镜提示词后出现在段块上，提示这段
      // 需要重渲（缓存键含 prompt，重渲时后端会自动重烧该段及其后所有段）。
      if (state.dirtySegs && state.dirtySegs.has(idx)) {
        const dm = h("div", "dirty", "✎");
        // ★ 第 27 轮：tooltip 清空 —— 原说明含"提示词"字样，一律不弹。
        //   状态本身仍可见：段块琥珀描边 .is-dirty + 右上 ✎ 角标。
        cell.appendChild(dm);
        cell.classList.add("is-dirty");
      }

      // A4：段块时长角标（新增时长），选中态（.is-plan-selected）下由 CSS 反白强调
      cell.appendChild(h("div", "dur", (seg.new_seconds || 0).toFixed(1) + "s"));

      // A5：段块勾选框（勾选=本次渲染这段；与选中=只看 区分）。写入 run_segments
      // 控件（后端按 "1,3,5-7" 解析），stopPropagation 避免触发段块选中/打开视频。
      //
      // ★ 第 36 轮：换成 <button type="button" role="checkbox" aria-checked>。
      //   原 <input type="checkbox"> 有几个坑：浏览器自动翻 checked +
      //   preventDefault 时机 + addEventListener click 的组合在不同 Chromium
      //   版本上行为不一致；用户报「勾不动 / 取消不了」就是这条链上的状态机
      //   没收敛。改成 button + aria-checked 后，状态完全自己掌控：点 → 我的
      //   handler 跑 → 算 next → setWidget + syncRunSegmentsToUI → aria-checked
      //   同步。没有浏览器原生 toggle，没有隐藏状态翻转，取消路径 100% 走我的代码。
      const chk = document.createElement("button");
      chk.type = "button";
      chk.setAttribute("role", "checkbox");
      chk.className = "run";
      // ★ 第 38 轮：集合语义 —— runSet 里有的段号 = 勾选。
      //   上一轮搞错成"起点/跟随"分层、把用户多选体验变成强制一串，
      //   用户反馈是想要"勾哪个就亮哪个"，不是"勾一个就自动亮一串"。
      //   后端会按 min(run_set) 自动选起点（首尾锚定耦合），我们这里只如实高亮用户的选择。
      const willRun = runSet.has(idx);
      chk.setAttribute("aria-checked", willRun ? "true" : "false");
      chk.dataset.run = willRun ? "on" : "";
      chk.title = willRun
        ? "已选：第 " + idx + " 段在本次渲染清单里。再点一次 = 取消（从清单移除）"
        : "加入本次渲染清单：把第 " + idx + " 段写进 run_segments 集合，"
          + "之后点工具栏「重渲已改段」会按这个集合触发。";
      cell.classList.toggle("is-run", willRun);

      // ★ 第 35 轮：抽出 syncRunSegmentsToUI —— widget 真值主导，所有改
      //   run_segments 的入口（段块 chk click + 工具栏"重渲已改段"按钮 + 后端
      //   回推）都走这一个出口同步视觉。否则工具栏按钮点了之后段块 chks 不动，
      //   用户以为"联动坏了"。chk click 和 toolbar 都直接调它。
      //
      // ★ 第 36 轮：chk 已是 <button> 而非 <input type="checkbox">，状态完全
      //   自己掌控 —— 没有浏览器原生 toggle，preventDefault 不再是必需的；
      //   但仍然留着以防万一（按下时也能阻断 form 提交等副作用）。
      chk.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        // ★ 第 38 轮：集合语义 —— 从 run_segments 集合里 toggle 当前段号，
        //   重新序列化为 "1,3,5-7" 写回 widget。勾哪个亮哪个，互不连带。
        //   后端 min(run_set) 会自动决定起点（首尾锚定耦合），与界面选择一致。
        const total = Number(segsRow.children.length) || 0;
        const set = parseRunSegments(readStr(node, "run_segments", ""), total);
        if (set.has(idx)) { set.delete(idx); }
        else { set.add(idx); }
        const serialized = Array.from(set).sort(function (a, b) { return a - b; }).join(",");
        setWidget(node, "run_segments", serialized);
        // 立即同步所有段块视觉（不等画布重绘）
        syncRunSegmentsToUI();
      });
      // 键盘：Space/Enter 切勾选（<button> 原生 click 已经会触发；这里留个
      //   keydown 兜底让 Enter/Space 都能切，preventDefault 防止页面滚动）。
      chk.addEventListener("keydown", (e) => {
        if (e.key === " " || e.key === "Enter") {
          e.preventDefault();
          chk.click();
        }
      });
      // ★ 第 39 轮：勾选框常驻（参考 ComfyUI_MiniMaxH3_Director 的 canvas 自绘位置）
      //   之前装进 .re-glass：hover 才显现，用户嫌麻烦（要"同时勾选多个不同分镜缩略图"得一个一个 hover）
      //   现在直接 cell.appendChild(chk)，绝对定位左下角 4px 常驻；
      //   .re-glass 只剩 ↻ 按钮，仍 hover 显现（防遮挡缩略图）。
      //   集合语义保持不变（toggle 段号 → Set → sort join），CSS/aria 与上轮一致。
      const reGlass = h("div", "re-glass");
      const bRe = h("button", "vact", "↻");
      bRe.title = "只重渲第 " + idx + " 段（首尾锚定，前段复用缓存）";
      bRe.onclick = (e) => { e.stopPropagation(); repairSegment(idx); };
      reGlass.appendChild(bRe);
      cell.appendChild(reGlass);
      // ★ 关键：chk 挂 cell 而不是 reGlass —— 由绝对定位 CSS 推到左下角常驻
      cell.appendChild(chk);

      // ── 缩略图与提示词一一对应 ────────────────────────────────────
      // 时间线原本只有画面，看不出这段讲什么。这里给每段常驻一行提示词摘要
      // （取该段最具信息量的字段，见 segBrief），并让整块可点选：
      // 选中后 .is-plan-selected 加深边框（样式见 CSS），提示词面板跟着联动。
      const _brief = segBrief(seg);
      // ★ 第 26 轮：段块上**完全不显示任何提示词文本**——
      //   之前的策略是「中文摘要画 .cap / 英文摘要写进 tooltip」，
      //   但用户明确要求「鼠标滑过禁止出现提示词」。
      //   hover 段块时画面只有 .re（"↻ 重渲该段"）按钮，干净。完整提示词
      //   在下面的「分镜提示词」面板里，那里才是正经入口。
      // ★ 第 27 轮：连 tooltip 也清空 —— 诊断文字也不再写进 cell.title。
      //   数据缺口信号只剩**红边 .is-noprompt**（纯视觉，不弹任何文字）。
      //   要查是哪段正文为空，看「分镜提示词」面板对应段（那里有明确的
      //   "本段没有解析到提示词正文" 占位，见 setPrompt 的 ta.placeholder）。
      if (!_brief) {
        cell.classList.add("is-noprompt");
      }
      cell.style.cursor = "pointer";
      cell.onclick = () => {
        // 走既有联动：state.planSelect 更新 state.planSelectedIndex 并触发
        // setPlanHighlight()，时间线与「功能规划」面板共用同一真相源，两边自动同步。
        // ★ 这里绝不能自行 add/remove .is-plan-selected —— 那会绕过真相源，
        //   让时间线高亮和面板选中各说各话。只在 planSelect 缺失时降级为纯视觉选中。
        if (state.planSelect) { state.planSelect(idx); return; }
        const prev = segsRow.querySelector(".h3d-seg.is-plan-selected");
        if (prev) prev.classList.remove("is-plan-selected");
        cell.classList.add("is-plan-selected");
      };
      // 借鉴融合：双击 = 新窗口打开完整视频；右键 = 段操作菜单。
      // mousedown 把焦点交回 segsRow，保证点完段块能直接用 ←→ 连续键盘操作。
      cell.onmousedown = () => {
        try { segsRow.focus({ preventScroll: true }); } catch (e) {}
      };
      cell.ondblclick = (e) => {
        e.stopPropagation();
        if (url) { try { window.open(url, "_blank"); } catch (err) {} }
      };
      cell.oncontextmenu = (e) => {
        // 第 26 轮：右键菜单整体移除 —— 仅 preventDefault 屏蔽浏览器原生菜单，
        // 不再触发任何东西。段块所有交互入口：单选 / 双击 / hover-.re。
        e.preventDefault();
        e.stopPropagation();
      };

      // hover 视频预览：浮层挂在 <body> 下（不是段块里）—— 面板长在节点内部，
      // 祖先容器带 overflow / transform，absolute 会被裁、fixed 会被带偏，
      // 只有挂到 body 才是真正的视口定位。坐标按段块的屏幕位置现算。
      if (url) {
        const pop = h("div", "h3d-pop is-body");
        const pv = document.createElement("video");
        pv.src = url;
        pv.muted = false;          // 悬停是"要看这一段"，给声音
        pv.loop = true;
        pv.playsInline = true;
        pv.controls = true;        // 悬停时也能拖进度 / 暂停
        pv.preload = "auto";
        pop.appendChild(pv);
        const meta = h("div", "meta",
          seg.id + " · 第 " + idx + " 段 · 新增 "
          + (seg.new_seconds || 0).toFixed(1) + "s");
        meta.appendChild(h("span", "tip", "点击段块 = 新窗口打开完整视频"));
        pop.appendChild(meta);
        document.body.appendChild(pop);
        cell.addEventListener("mouseenter", () => {
          const r = cell.getBoundingClientRect();
          // 浮层默认向正上方弹；上方顶到屏幕外就翻到段块下方
          const above = r.top - 8;
          const flip = above < 240;
          pop.style.left = Math.round(r.left + r.width / 2) + "px";
          pop.style.top = Math.round(flip ? r.bottom + 8 : above) + "px";
          pop.style.transform = flip
            ? "translate(-50%, 0)" : "translate(-50%, -100%)";
          pop.classList.add("is-on");
          try { pv.currentTime = 0; pv.play().catch(() => {}); } catch (e) {}
        });
        cell.addEventListener("mouseleave", () => {
          pop.classList.remove("is-on");
          try { pv.pause(); } catch (e) {}
        });
      }

      // 有视频 → 切换「功能规划」版块到这一段（plan 面板右栏会
      // 渲染带控件的 video，左栏显示对应提示词）；
      // 没渲染 → 同上 + 右栏显示首帧占位图。
      // 老逻辑「打开完整视频」改成 hover 弹层（仍在）+ 「功能规划」右栏
      // 的 controls；「单段重渲」改成 hover 弹出来的"↻ 重渲该段"按钮。
      cell.onclick = () => state.planSelect && state.planSelect(idx);
      segsRow.appendChild(cell);

      // 轨道：暗虚线打底 + 亮实线填充（小马跑到哪儿填到哪儿，与段块同宽）
      const rl = h("div", "h3d-rail-seg");
      rl.style.flexGrow = String(grow);
      rl.appendChild(h("i", "base"));
      rl.appendChild(h("i", "fill"));
      railLine.appendChild(rl);
      railSpans[idx - 1] = rl;
    });

    const total = segs.reduce((a, s) => a + (s.new_seconds || 0), 0);
    const pct = Math.round((doneSet.size / segs.length) * 100);
    barFill.style.width = pct + "%";
    // 数字自解释 —— 不加前缀 "4 / 6" 容易被读成"当前在第 4 段（共 6）"，
    // 跟状态栏里"第 5/6 段"撞到一起。明确写"已渲"后与状态栏分工清晰。
    // ★ 第 40 轮：这里的 total 只累加 new_seconds（锚定尾巴是重放，成片里
    //   不算新内容），所以标尺末端的"总时长"会比这里大一截（多了各段
    //   handoff）。加"成片"二字，两个数字就不会被当成互相矛盾。
    barText.textContent = "已渲 " + doneSet.size + " / " + segs.length
      + " · 成片 " + total.toFixed(1) + "s";
    tag.textContent = doneSet.size === segs.length ? "✓ 完成"
      : "已渲 " + doneSet.size + "/" + segs.length;
    tag.className = "h3d-tag "
      + (doneSet.size === segs.length ? "is-ok" : "");

    // 轨道重画：已渲=亮实线铺满，当前段交给小马那边的 rAF 按进度填，
    // 未渲=暗虚线。跑着的时候别抢 rAF 的位置，只画到 0。
    if (_running) paintRail(0); else setPony(false);
    paintRuler();   // A2：标尺刻度随段数 / 宽度重排
    refit(node);   // 段数/进度变了，时间线块会换行，节点高度跟着贴合
  }

  /* ★ 段完成即刷新（第 20 轮）：后端 segment_done 事件**不带**视频 url，
     前端必须重拉 /h3/session 才知道这一段已经落盘 —— 时间线才能把参考图
     换成真实的分段视频。原来只在「最后一段完成」时才拉，于是：
       · 整条渲染跑到一半时，已完成的段仍挂着参考图；
       · 单段修复（total=1 之外的情形）压根不触发。
     多次事件合并成一次请求 —— 只做合流、不设上限，保证最后一次一定发出去。 */
  let _refreshTimer = null;
  function scheduleRefreshRendered(delay) {
    if (_refreshTimer) clearTimeout(_refreshTimer);
    _refreshTimer = setTimeout(() => {
      _refreshTimer = null;
      refreshRendered();
    }, typeof delay === "number" ? delay : 400);
  }

  async function refreshRendered() {
    try {
      const res = await api.fetchApi(
        "/h3/session?name=" + encodeURIComponent(sessionName()));
      const data = await res.json();
      state.rendered.clear();
      (data.segments || []).forEach((s) => {
        if (s.url) state.rendered.set(s.index, s.url + "&t=" + Date.now());
      });
      doneSet = new Set((data.segments || []).map((s) => s.index));
      state.done = Number(data.done || 0);
      render();
    } catch (e) { /* 会话还不存在，正常 */ }
  }

  /* 单段修复：直接驱动本 Director 的 repair_segment 控件，只重渲那一段。
     复用后端 H3RepairSegmentNode 的首尾锚定逻辑，写回同一会话并更新 manifest，
     再走统一的拼接 / 时间线收尾。跑完自动复位为 0，不影响下次整条渲染。 */
  function repairSegment(index) {
    const w = findWidget(node, "repair_segment");
    if (!w) {
      setPony(false, "⚠ 本 Director 没有「单段修复」控件，请更新 H3Director 节点");
      return;
    }
    const sess = sessionName();
    w.value = index;
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    setPony(true, "⏳ 已排队：重渲第 " + index + " 段（" + sess + "）");
    try {
      app.queuePrompt(0, 1);
    } catch (e) {
      setPony(false, "⚠ 排队失败：" + (e && e.message ? e.message : e));
    }
    setTimeout(() => {
      // 还原：避免下一次整条渲染时只渲了一段
      if (w && w.value === index) w.value = 0;
      if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, 2000);
  }

  // 全部渲染：force-all（resume=false）排队
  btnFull.onclick = (e) => {
    e.stopPropagation();
    const w = findWidget(node, "resume");
    const oldVal = w ? w.value : true;
    if (w) w.value = false;
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    setPony(true, "▶ 已排队：全部渲染（忽略缓存，整条重渲）");
    try { app.queuePrompt(0, 1); }
    catch (err) { setPony(false, "⚠ 排队失败：" + (err && err.message ? err.message : err)); }
    setTimeout(() => {
      if (w && w.value === false) w.value = true;     // 复位成安全的「断点」模式
      if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, 2000);
  };
  /* 断点渲染：先问后端「这个会话在磁盘上渲到第几段」，再按 resume=true 排队。
     进度取自 output\h3_continuous\<会话>\ 下真实存在的 seg_NN.mp4，而不是
     manifest —— manifest 可能比磁盘落后（少记了已经渲好的段），后端这次会
     按磁盘补齐记录，那些段就不再重烧。 */
  btnResume.onclick = async (e) => {
    e.stopPropagation();
    let done = 0;
    try {
      const res = await api.fetchApi(
        "/h3/session?name=" + encodeURIComponent(sessionName()));
      const data = await res.json();
      done = Number(data.done || 0);
      state.done = done;
      (data.segments || []).forEach((s) => {
        if (s.url) state.rendered.set(s.index, s.url + "&t=" + Date.now());
      });
      doneSet = new Set((data.segments || []).map((s) => s.index));
      render();
    } catch (err) { /* 会话还不存在：从第 1 段开始，正常 */ }

    const total = (state.segments() || []).length || 0;
    const w = findWidget(node, "resume");
    if (w) w.value = true;
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    setPony(true, done > 0
      ? "↻ 断点渲染：已识别 " + done + " 段，从第 " + (done + 1) + " 段继续"
        + (total ? "（共 " + total + " 段）" : "")
      : "↻ 断点渲染：磁盘上没有已完成的段，从第 1 段开始");
    try { app.queuePrompt(0, 1); }
    catch (err) { setPony(false, "⚠ 排队失败：" + (err && err.message ? err.message : err)); }
    setTimeout(() => {
      // 断点本来就是默认；万一用户事先把它关过，这里还原
      if (w && w.value !== true) w.value = true;
      if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, 2000);
  };

  /* 一键重渲「提示词已改」的段：把 dirty 段号全部写进 run_segments 集合
     （不是只写一个起点）。后端按 min(run_set) 从最早改过的段起重渲到结尾，
     链式缓存键让 N 之后也跟着失效。勾哪个亮哪个（集合语义）。 */
  btnRerunDirty.onclick = (e) => {
    e.stopPropagation();
    const dirty = Array.from(state.dirtySegs || []).filter((i) => i > 0);
    if (!dirty.length) {
      setPony(false, "没有「已改待重渲」的段 —— 先在分镜提示词框里改内容");
      return;
    }
    const sorted = dirty.slice().sort(function (a, b) { return a - b; });
    const total = (state.segments() || []).length || 0;
    // ★ 第 38 轮：写入完整集合（不是只写 min），段块视觉立即同步显示勾选状态。
    setWidget(node, "run_segments", sorted.join(","));
    syncRunSegmentsToUI();
    const w = findWidget(node, "resume");
    if (w) w.value = true;          // 未改的前段复用磁盘缓存
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    setPony(true, "↻ 已排队：勾选段 " + sorted.join(",") + " 起重渲"
      + "（已改 " + sorted.length + " 段"
      + (total ? "，共 " + total + " 段" : "")
      + "；后端按最早段号起到结尾）");
    try { app.queuePrompt(0, 1); }
    catch (err) { setPony(false, "⚠ 排队失败：" + (err && err.message ? err.message : err)); }
    setTimeout(() => {
      if (w && w.value !== true) w.value = true;
      if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }, 2000);
  };

  btnRefresh.onclick = (e) => { e.stopPropagation(); refreshRendered(); };

  /* Director 进度推送 */
  function timelineProgressHandler(ev) {
    const d = ev.detail || {};
    if (String(d.node_id) !== String(node.id)) return;
    if (d.event === "plan") {
      // 后端会把磁盘上已有的段数（done_count）一起推过来。断点模式下如果
      // 只清零 doneSet，进度条会从「已渲 4/6」瞬间掉回「已渲 0/6」再慢慢
      // 往上爬 —— 而且爬得还不对（被清掉的 4 段永远不会被加回来）。
      // 优先用后端下发的真实段号列表（含中间缺段时后面的孤立段），这样跟
      // /h3/session 的 segments 口径一致，刷新面板时「已渲 X/Y」不会跳变。
      // 老后端没这个字段时退回 1..done_count。
      if (d.done_indices && d.done_indices.length) {
        doneSet = new Set(d.done_indices);
      } else {
        doneSet = new Set();
        const n = Number(d.done_count || 0);
        for (let i = 1; i <= n; i++) doneSet.add(i);
      }
      activeIndex = -1;
      render();
      setPony(true, "准备渲染 " + d.total + " 段…");
    } else if (d.event === "segment_start") {
      // 这一段正在渲，就不再算「已渲」。否则重渲一个磁盘上已存在的段时，
      // 进度条会把它同时算成已完成（「已渲 2/4」）又在状态栏写「正在渲第
      // 1/4 段」——同一个段既是"已完成"又是"正在渲"，自相矛盾。
      // 等 segment_done 再把它加回 doneSet，计数自然对得上。
      doneSet.delete(d.index);
      activeIndex = d.index;
      render();
      setPony(true, "第 " + d.index + "/" + d.total + " 段 · " + d.shot_id
        + " · " + d.frames + " 帧 · " + (d.task || "")
        + " · 参考图 " + d.refs + " 张");
    } else if (d.event === "segment_done") {
      doneSet.add(d.index);
      // 该段已用新提示词重渲完成 → 清掉「已改待重渲」角标
      if (state.dirtySegs) state.dirtySegs.delete(d.index);
      activeIndex = -1;
      render();
      if (d.index < d.total) {
        setPony(true, "✓ 第 " + d.index + "/" + d.total + " 段完成（"
          + d.elapsed + "s）");
      } else {
        setPony(false, "✓ 全部完成（共 " + d.total + " 段）");
      }
      // ★ 每段完成都拉一次 /h3/session（第 20 轮）：只有这样才能拿到该段
      //   刚落盘的 seg_NN.mp4，把时间线上的参考图换成真实分段视频。
      //   末段多留 1.2s 给后端收尾（拼接 / 写 manifest）。
      scheduleRefreshRendered(d.index >= d.total ? 1200 : 400);
    } else if (d.event === "finish") {
      activeIndex = -1;
      render();
      setPony(false, "✓ 全部完成");
      refreshRendered();
    }
  }
  api.addEventListener("h3.director.progress", timelineProgressHandler);

  function windowResizeHandler() { if (!_running) setPony(false); }
  if (typeof window !== "undefined") {
    window.addEventListener("resize", windowResizeHandler);
  }

  // ★ 小马"飞出节点"的根因：小马静止（未在渲染）时，它的 left 像素坐标
  //   是 setPony(false) 那一刻按 span()（= rail.clientWidth - PONY_W）算好
  //   就定死的，没有任何东西会在之后重新贴合。以前唯一会重算的时机是
  //   window 的 resize 事件——但拖节点边缘、折叠/展开版块都是画布内部操作，
  //   根本不会触发浏览器 window 的 resize，于是小马停在"按旧的、更宽的
  //   轨道宽度"算出来的坐标上；节点被拖窄之后，这个坐标落在新轨道之外，
  //   由于 rail 原来没裁切，小马就视觉上"飞"到了节点右侧的空白画布里
  //   （overflow:hidden 是兜底，这里才是真正让坐标重新跟上轨道宽度）。
  //   直接观察 rail 自身尺寸变化，不管是谁引起的宽度变化都能重新贴合。
  let _railRO = null;
  try {
    if (typeof ResizeObserver !== "undefined") {
      _railRO = new ResizeObserver(() => { if (!_running) { setPony(false); paintRuler(); } });
      _railRO.observe(rail);
    }
  } catch (e) { /* 不支持就算了 */ }

  function cleanupTimeline() {
    try { api.removeEventListener("h3.director.progress", timelineProgressHandler); } catch (e) { /* 不支持移除就算了 */ }
    try { window.removeEventListener("resize", windowResizeHandler); } catch (e) { /* 不支持移除就算了 */ }
    try { if (_railRO) _railRO.disconnect(); } catch (e) { /* 已断开就算了 */ }
    if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    // 面板销毁后别再往 /h3/session 打请求（节点删除 / 折叠重建时走这里）
    if (_refreshTimer) { clearTimeout(_refreshTimer); _refreshTimer = null; }
    _running = false;
  }

  /** 同步「功能规划」选中态到时间线缩略图（轻量，不重建 DOM）。
   *  在 mountPanels 里被 state.planSelect 调一下，状态栏那边变了
   *  这边就能立刻高亮当前段，不重新跑 render() 的 innerHTML 整段清空。 */
  function setPlanHighlight() {
    const sel = Number(state.planSelectedIndex) || 0;
    const cells = segsRow.querySelectorAll(".h3d-seg");
    cells.forEach((c) => {
      const idx = Number(c.dataset.index);
      if (idx === sel && sel > 0) c.classList.add("is-plan-selected");
      else c.classList.remove("is-plan-selected");
    });
  }

  sec.appendChild(hd);
  sec.appendChild(bd);
  // bd 一并暴露：「功能规划」的内容要挂进来，才能跟着本版块一起折叠。
  // refreshDirty：提示词被改后由 state.onDirty 回调，重绘段块刷新「✎ 已改」角标。
  return { sec, bd, badge: head.badge, render, refreshRendered, setPlanHighlight,
           refreshDirty: render, cleanup: cleanupTimeline };
}

/* ==================================================================== */
/* 分镜提示词编辑：模块级共用逻辑                                        */
/* -------------------------------------------------------------------- */
/* 「从 PACK 原文切段 / 整段替换 / 时间标注 / 写回」这套函数原来都埋在     */
/* buildPlanPanel 闭包里，只有 H3Director 的面板能用。现在提到模块级，    */
/* 让 H3PromptPackParser 上的编辑器也能复用同一份实现 —— 原来那套独立的   */
/* web/js/h3_shot_editor.js 已停用（能力并入本文件），避免两个文件各挂    */
/* 一套面板互相打架。                                                    */
/* ==================================================================== */
const H3_FIELD_ORDER = ["subject_definitions", "summary", "retention_analysis",
  "detailed_description", "overall_soundscape", "non_diegetic_music"];

/** 后端 /h3/pack_preview 的 issues 里，哪些算「这段已经废了」需要红字告警。
 *  判据对应后端 routes._pack_preview 追加的那条「整段正文为空」，以及
 *  prompt_pack 自己报的「缺少 detailed_description 字段」。
 *  ★ 第 30 轮新增：以前 issues 整条不展示，坏 PACK 全程静默。 */
const H3_FATAL_ISSUE_RE = /(正文为空|缺少\s*detailed_description|没有解析到任何分镜)/;

/** 找画布上的 H3PromptPackParser 节点（工作流里就是 720）。 */
function h3FindParser() {
  const nodes = graphNodes();
  return nodes.find((n) => n.comfyClass === "H3PromptPackParser"
    || n.type === "H3PromptPackParser") || null;
}

/** 当前 PACK 源文本：H3PromptPackParser 的 pack_text 上游文本框（如 361）。 */
function h3PackText() {
  const parser = h3FindParser();
  if (!parser) return "";
  const src = upstream(parser, "pack_text");
  if (!src) return "";
  const w = (src.widgets || []).find(
    (x) => x && typeof x.value === "string" && x.value.length > 20);
  if (w) return w.value;
  const dom = (src.widgets || []).find(
    (x) => x && x.element && x.element.querySelector);
  if (dom) {
    const ta = dom.element.querySelector("textarea");
    if (ta) return ta.value;
  }
  return "";
}

/** 把一段的六字段拼成标准六段式正文（与后端 _format_segment_fields 同格式：
 *  字段名行 + 内容，字段间空行）。fields 为空时退回 fallback 正文。 */
function h3SegToBlock(fields, fallback) {
  const parts = [];
  H3_FIELD_ORDER.forEach((f) => {
    const v = fields ? fields[f] : "";
    if (v && String(v).trim()) {
      parts.push(f + ":\n" + String(v).replace(/\n+$/, ""));
    }
  });
  if (parts.length) return parts.join("\n\n");
  return String(fallback || "").trim();
}

/** 定位「某段某语言版」在 PACK 行数组里的字段区范围 [start, end)（不含段头行）。
 *  返回 null 表示没找到。双语 PACK 按 lang 限定在 [1] 中文版 / [2] 英文版 版块内。 */
function h3LocateSegRange(lines, segId, lang) {
  const num = (s) => { const m = String(s || "").match(/\d+/); return m ? m[0] : ""; };
  const target = num(segId);
  if (!target) return null;
  let lo = 0, hi = lines.length;
  const marks = [];
  lines.forEach((ln, i) => {
    const m = ln.trim().match(/^\[\s*([12])\s*\]\s*(.*)$/);
    if (!m) return;
    const name = m[2] || "";
    const l = /中文/.test(name) ? "zh"
      : (/英文/.test(name) || /\ben\b/i.test(name)) ? "en" : null;
    if (l) marks.push([i, l]);
  });
  if (lang === "zh" || lang === "en") {
    for (let k = 0; k < marks.length; k++) {
      if (marks[k][1] === lang) {
        lo = marks[k][0] + 1;
        hi = (k + 1 < marks.length) ? marks[k + 1][0] : lines.length;
        break;
      }
    }
  }
  const segRe = /^\s*#{3,}\s*(S?\d+[A-Za-z]?)\s*\/.*?#{3,}\s*$/;
  let start = -1, end = hi;
  for (let i = lo; i < hi; i++) {
    const m = lines[i].trim().match(segRe);
    if (!m) continue;
    if (start < 0) {
      if (num(m[1]) === target) start = i + 1;
    } else {
      end = i;
      break;
    }
  }
  if (start < 0) return null;
  for (let i = start; i < end; i++) {
    const s = lines[i].trim();
    if (/^={5,}$/.test(s) || /^END\s+OF\s+PROMPTS$/i.test(s)) { end = i; break; }
  }
  return { start, end };
}

/** 从 PACK 原文切出「某段某语言版」的六段式正文。不依赖后端 full_fields ——
 *  改 JS 立即生效，且显示的就是 PACK 原文，保真度最高。 */
function h3ExtractSegBlock(packText, segId, lang) {
  const text = String(packText || "");
  if (!text) return "";
  const lines = text.split("\n");
  const r = h3LocateSegRange(lines, segId, lang);
  if (!r) return "";
  return lines.slice(r.start, r.end).join("\n").trim();
}

/** 用编辑后的整段正文替换 PACK 里对应段的字段区（纯前端字符串操作）。 */
function h3ReplaceSegBlock(packText, segId, lang, newBody) {
  const text = String(packText || "");
  if (!text) return text;
  const lines = text.split("\n");
  const r = h3LocateSegRange(lines, segId, lang);
  if (!r) return text;
  const bodyLines = String(newBody || "").replace(/\n+$/, "").split("\n");
  const out = lines.slice(0, r.start).concat([""], bodyLines, [""], lines.slice(r.end));
  return out.join("\n");
}

/** 分镜时间统计：取 PACK 段头 ########## S02 / 11s+1.6=12.6 / EN ##########
 *  里 `/` 之间的时长标注（与段头同源）。取不到返回空串。 */
function h3SegTimeLabel(packText, segId, lang) {
  const text = String(packText || "");
  if (!text) return "";
  const lines = text.split("\n");
  const r = h3LocateSegRange(lines, segId, lang);
  if (!r || r.start <= 0) return "";
  const head = lines[r.start - 1] || "";
  const m = head.match(
    /^\s*#{3,}\s*S?\d+[A-Za-z]?\s*\/\s*(.*?)\s*(?:\/.*)?\s*#{3,}\s*$/);
  return m ? String(m[1]).trim() : "";
}

/** 把整份 PACK 写回上游文本框（PrimitiveStringMultiline，如 361）。
 *  ★ 四路同步：只改 DOM textarea 而不改 widget.value 的话，Queue 时节点仍读旧值。 */
function h3WritePackSource(text) {
  const parser = h3FindParser();
  if (!parser) return;
  const src = upstream(parser, "pack_text");
  if (!src) return;
  (src.widgets || []).forEach((w) => {
    if (!w) return;
    const ta = (w.element && w.element.querySelector)
      ? w.element.querySelector("textarea") : null;
    if (typeof w.value === "string" && w.value !== text) w.value = text;
    if (ta && ta.value !== text) ta.value = text;
    if (typeof w.callback === "function") {
      try { w.callback(text, app && app.canvas, src, [0, 0], null); }
      catch (e) { /* callback 签名不同，失败不影响 */ }
    }
    if (ta) {
      try { ta.dispatchEvent(new Event("input", { bubbles: true })); }
      catch (e) { /* 忽略 */ }
    }
  });
  try { src.setDirtyCanvas(true, true); } catch (e) { /* 老版本没有 */ }
}

/** 把整份 PACK 写进 H3PromptPackParser 的 pack_override（覆盖通道）：
 *  execute 里 pack_override 非空时优先用它，不污染外侧 PACK 文本框。
 *  值必须是"看起来像 PACK"的文本，否则后端守卫会忽略（见 pack_nodes.py）。 */
function h3WritePackOverride(text) {
  const parser = h3FindParser();
  if (!parser) return;
  const w = (parser.widgets || []).find((x) => x && x.name === "pack_override");
  if (!w) return;
  if (typeof w.value === "string" && w.value !== text) w.value = text;
  const ta = (w.element && w.element.querySelector)
    ? w.element.querySelector("textarea") : null;
  if (ta && ta.value !== text) ta.value = text;
  if (typeof w.callback === "function") {
    try { w.callback(text, app && app.canvas, parser, [0, 0], null); }
    catch (e) { /* 忽略 */ }
  }
  if (ta) {
    try { ta.dispatchEvent(new Event("input", { bubbles: true })); }
    catch (e) { /* 忽略 */ }
  }
  try { parser.setDirtyCanvas(true, true); } catch (e) { /* 忽略 */ }
}

/* ------------------------------------------------------------------ */
/* 版块：功能规划 —— 左栏分镜提示词 + 右栏对应分镜预览（挂在时间线 bd 内）。 */
/* ------------------------------------------------------------------ */
/* 点击时间线缩略图 / 渲染中自动切换会更新本版块：左栏 prompt（来自
 * pack_preview 的 detailed_description），右栏预览（已渲=<video>，未渲=首帧）。
 * 选中态由 state.planSelectedIndex 持有，时间线 .is-plan-selected 呼应。 */
function buildPlanPanel(node, state) {
  // ★ 不再自建 .h3d-sec / 序号徽标 /「功能规划」标题：整块内容由 mountPanels
  //   挂进「8 分镜时间线」的 .h3d-bd，折叠时间线时一起收起（用户要求）。
  //   只保留一行小标题 + 「重选」按钮，说明这块是时间线的附属视图。
  const body = h("div", "h3d-plan-inline");
  const barTop = h("div", "h3d-plan-bar");
  // parser 上这里展示的是「剧本 → 分镜」转换后的完整提示词，标题随之改口径
  const parserMode = isParserNode(node);
  barTop.appendChild(h("span", "h3d-plan-bar-lb",
    parserMode ? "剧本转换分镜 · 完整分镜提示词" : "分镜提示词 / 预览"));

  /* ★ 2026-09-19：语言 / 写回模式工具栏的**落点**（DOM 在下面才建）。
     插在标题之后、重选之前，于是「标题 · 语言组 · 模式组 · 重选」同处一行。
     此前 .h3d-plan-tools 是 grid 之上独立的一行（body.appendChild(toolRow)），
     结果「重选」孤挂在标题行右端、与下方 4 个胶囊错开一行 —— 用户截图
     反馈的「按钮没水平对齐」就是这个。 */
  const toolSlot = h("span", "h3d-plan-tools");
  barTop.appendChild(toolSlot);

  // parser 上没有时间线缩略图，给一排段按钮切换查看哪一段（Director 用缩略图切）。
  const segBtns = parserMode ? h("span", "h3d-seg-btns") : null;
  if (segBtns) {
    segBtns.style.cssText = "display:inline-flex;gap:4px;flex-wrap:wrap;margin-left:auto;";
    barTop.appendChild(segBtns);
  }
  const btnReset = h("button", "h3d-btn", "重选");
  if (!parserMode) barTop.appendChild(btnReset);   // parser 上没有「未选」态，重选无意义
  body.appendChild(barTop);

  // 两栏布局
  const grid = h("div", "h3d-plan-grid");

  // ===== 左栏：分镜提示词（整段六段式实时编辑） =====
  // 借鉴工作流 bilingual-pack 逻辑：英文版 EN 给 Director 直接渲染，中文版 ZH
  // 作为对照源 / 可编辑后写回中文版块。每段把六段式正文（subject_definitions
  // … non_diegetic_music）整体放进**一个**文本框，编辑后通过 /h3/pack_apply_block
  // 原样整块回写整份 PACK（rebuild_pack_block 按 lang 命中对应版块）。
  let edLang = "en";              // 当前编辑语言侧（en / zh）
  let editorEls = null;           // {en: textarea, zh: textarea}
  /* ★ 当前文本框里装的到底是「哪一段 / 哪一语言」的内容。
     没有这个身份标记，就会出这种事（实测把用户的英文版整段清空）：
       1. 选中 S01（它在 PACK 里的英文块是空的）→ h3ExtractSegBlock 返回 ""，
          于是 3583 行兜底用 full_fields 拼出「自动修正模板」填进文本框；
       2. 焦点还在文本框里（activeElement 守卫拦住重填），用户点 S04；
       3. doWrite 拿到的 body 其实是 **S01 的模板**，而 current 是 **S04 的真身**，
          两者不等 → 判定「用户改了内容」→ 把 S04 的 33 行英文块覆写成 1 个字段。
     写入前必须比对身份：只有「文本框装的就是当前 (段号,语言)」才允许回写。 */
  let editorLoaded = { seg: 0, lang: "", fromFallback: false };
  /* 最近一次 setPrompt 渲染的段号（供切语言时重填用）。 */
  const loadedRef = { current: 0 };

  const cardPrompt = h("div", "h3d-plan-card");
  const hdPrompt = h("div", "h3d-plan-hd");
  hdPrompt.appendChild(h("span", "lb", "提示词"));
  const who = h("span", "who");
  who.appendChild(document.createTextNode("—"));
  hdPrompt.appendChild(who);
  hdPrompt.appendChild(h("span", "stat", "未选"));
  cardPrompt.appendChild(hdPrompt);

  // 语言切换标签：EN=Director 直接渲染；ZH=对照源/编辑后写回中文版块
  const tabBar = h("div", "h3d-lang-tabs");
  const tabEn = h("button", "h3d-lang-tab is-active", "英文 EN");
  const tabZh = h("button", "h3d-lang-tab", "中文 ZH");
  tabBar.appendChild(tabEn);
  tabBar.appendChild(tabZh);
  tabEn.onclick = () => { edLang = "en"; tabEn.classList.add("is-active"); tabZh.classList.remove("is-active"); applyLang(); };
  tabZh.onclick = () => { edLang = "zh"; tabZh.classList.add("is-active"); tabEn.classList.remove("is-active"); applyLang(); };
  // 语言 + 写回模式两组胶囊挂进标题行的 toolSlot（见上，barTop 已 append 到 body）。
  // 仍然**不进左栏卡片**：那样左栏内容区会被推低、右栏视频却紧贴标题，左右上下都错位。
  toolSlot.appendChild(tabBar);

  // 写回模式切换（吸收原 h3_shot_editor.js 的「写回PACK文本框 / 覆盖Parser」）：
  //   pack     —— 重拼后的整份 PACK 写回外侧文本框，所见即所得（默认）
  //   override —— 写进 parser 的 pack_override，不动原文，execute 优先读它
  const modeBar = h("div", "h3d-lang-tabs");
  const modePack = h("button", "h3d-lang-tab is-active", "写回PACK");
  const modeOv = h("button", "h3d-lang-tab", "覆盖Parser");
  modePack.title = "编辑后重拼整份 PACK，写回外侧 PACK 文本框（所见即所得，默认）";
  modeOv.title = "编辑后写入 H3PromptPackParser 的 pack_override —— 不改原文，"
    + "execute 优先读它；适合只想临时试提示词、不想动源文本框的场合";
  modeBar.appendChild(modePack);
  modeBar.appendChild(modeOv);
  const applyMode = () => {
    modePack.classList.toggle("is-active", state.writeMode !== "override");
    modeOv.classList.toggle("is-active", state.writeMode === "override");
  };
  modePack.onclick = () => { state.writeMode = "pack"; applyMode(); };
  modeOv.onclick = () => { state.writeMode = "override"; applyMode(); };
  toolSlot.appendChild(modeBar);
  // 工具栏已并入 barTop（body.appendChild(barTop) 在上方完成），不再单独占一行。
  applyMode();

  const bdPrompt = h("div", "h3d-plan-bd is-empty h3d-ed");
  bdPrompt.textContent = "还没解析 PACK，或者还没选中分镜。";
  cardPrompt.appendChild(bdPrompt);
  grid.appendChild(cardPrompt);

  // 首次构建编辑区：EN/ZH 各一个 textarea（各自装整段六段式），按 lang 显隐。
  function buildEditor() {
    bdPrompt.className = "h3d-plan-bd h3d-ed";
    bdPrompt.textContent = "";
    editorEls = {};
    [["en", "英文版六段式提示词（subject_definitions: … non_diegetic_music:）"],
     ["zh", "中文版六段式提示词（subject_definitions: … non_diegetic_music:）"]]
      .forEach(([lang, ph]) => {
        const ta = document.createElement("textarea");
        ta.className = "h3d-block-ta is-" + lang;
        ta.dataset.lang = lang;
        ta.placeholder = ph;
        ta.spellcheck = false;
        ta.addEventListener("input", scheduleWrite);
        bdPrompt.appendChild(ta);
        editorEls[lang] = ta;
      });
    applyLang();
  }

  function applyLang() {
    if (!editorEls) return;
    editorEls.en.style.display = edLang === "en" ? "" : "none";
    editorEls.zh.style.display = edLang === "zh" ? "" : "none";
    /* ★ 切语言 = 换了一份正文。必须把新语言的文本框重新按当前段号填一遍，
       并刷新身份标记。否则旧语言残留的内容会在下一次 doWrite 时被写进
       新语言的版块（把英文块写成中文内容，或反向）。
       refillOnly 防止 setPrompt → applyLang → setPrompt 递归。 */
    if (loadedRef.current && !applyLang.refilling) {
      applyLang.refilling = true;
      try { setPrompt(loadedRef.current); }
      finally { applyLang.refilling = false; }
    }
  }

  // 编辑防抖写回：停手 800ms 才回写整份 PACK，避免每键一次重解析。
  let writeTimer = 0;
  /* ★ 防抖期间若用户切了段 / 切了语言，这次待写的编辑就属于「上一段」,
     必须丢弃 —— 否则 800ms 后 doWrite 会拿旧内容配新段号。这里记下
     排队时那一刻的身份，doWrite 再核一次。 */
  let writeTicket = null;
  function scheduleWrite() {
    writeTicket = { seg: state.planSelectedIndex, lang: edLang };
    if (writeTimer) clearTimeout(writeTimer);
    writeTimer = setTimeout(() => { writeTimer = 0; doWrite(); writeTicket = null; }, 800);
  }

  function doWrite() {
    const idx = state.planSelectedIndex;
    if (!idx || idx <= 0) return;
    /* ★ 排队时与落笔时的身份必须一致（防抖期间用户切段 → 丢弃这次待写）。 */
    if (writeTicket && (writeTicket.seg !== idx || writeTicket.lang !== edLang)) return;
    const segs = state.segments();
    const seg = segs[idx - 1];
    if (!seg || !editorEls) return;
    /* ★ 写入前身份校验（防「清空英文版」事故，见 editorLoaded 注释）。
       ① 文本框装的内容必须就是当前 (段号, 语言)。焦点还在文本框里时，
          setPrompt 不会重填，用户点别的段 → 这里的 seg 已变、文本框没变，
          不拦就会把上一段的正文写到这一段上。
       ② 兜底拼出来的正文（PACK 里本来没这一块）绝不能回写 —— 那会把后端
          自动修正生成的模板当成用户输入，覆盖 PACK 原文。 */
    if (editorLoaded.seg !== idx || editorLoaded.lang !== edLang) return;
    if (editorLoaded.fromFallback) return;
    const body = editorEls[edLang].value;
    const packText = state.packText();
    if (!packText) return;
    // 内容没真变就不写（用户只是聚焦/失焦、或改回原样），避免误标「已改」。
    const current = h3ExtractSegBlock(packText, seg.id, edLang);
    if (body.trim() === current.trim()) return;
    // ★ 空正文 + 原文非空 = 误伤（文本框还没填上就被判定为「用户删光了」）。
    //   真要清空请手动全选删除 —— 那种情况下 body 会带用户的操作痕迹，
    //   这里只拦「写入瞬间 body 为空、而磁盘上原文非空」这种明显的时序错位。
    if (!body.trim() && current.trim()) return;
    // 纯前端替换该段正文 → 写回 PACK 源（同步 widget.value）→ 触发 input →
    // 重解析 → 面板刷新。不经过后端，改 JS 后刷新浏览器即可生效。
    const next = h3ReplaceSegBlock(packText, seg.id, edLang, body);
    if (next !== packText) {
      // 写回模式（面板右上角切换）：
      //   pack     —— 写回外侧 PACK 文本框（所见即所得，默认）
      //   override —— 写进 parser 的 pack_override（不动原文，execute 优先读它）
      if (state.writeMode === "override") h3WritePackOverride(next);
      else h3WritePackSource(next);
      // 联动：标记该段提示词已改 → 时间线打「✎ 已改」角标 + 可一键重渲
      if (state.onDirty) state.onDirty(idx);
    }
  }

  // ===== 右栏：分镜预览 =====
  // ===== 右栏：分镜预览（仅 H3Director） =====
  // parser 节点不渲染，没有视频 / 首帧可预览，所以右栏整块不建 —— 让「剧本转换
  // 分镜后的完整提示词」独占整行，展示空间更宽。
  const showPreview = !parserMode;
  let cardPrev = null, hdPrev = null, whoPrev = null, bdPrev = null;
  if (showPreview) {
    cardPrev = h("div", "h3d-plan-card");
    hdPrev = h("div", "h3d-plan-hd");
    hdPrev.appendChild(h("span", "lb", "分镜预览"));
    whoPrev = h("span", "who");
    whoPrev.appendChild(document.createTextNode("—"));
    hdPrev.appendChild(whoPrev);
    hdPrev.appendChild(h("span", "stat", "未选"));
    cardPrev.appendChild(hdPrev);
    bdPrev = h("div", "h3d-plan-bd is-empty");
    bdPrev.textContent = "选中分镜后这里会显示对应视频 / 首帧。";
    cardPrev.appendChild(bdPrev);
    grid.appendChild(cardPrev);
  }

  body.appendChild(grid);
  // 折叠交给「8 分镜时间线」的 collapsible —— 这里不能再挂一次，
  // 否则点时间线标题只会收起轨道、这块还留着。

  // 占位图查找：本段引用的槽位 → 该 slot 上挂的参考图
  // ★ 第 19 轮：与时间线那处 refThumbFor **口径完全一致**（两处不在同一闭包，
  //   改动必须同步，否则"时间线对了、分栏还是老样子"= 改一半）。
  function refThumbFor(seg) {
    if (!seg) return "";
    const slots = refSlotImages(node);
    const pick = (n) => slots.find((s) => s.slot === n && s.file);
    // 作者填的分类元数据（{slot,kind,name}）—— 和时间线那处同源，决定"人/非人"
    const metas = localRefMetas();
    const perShot = perShotPicSlots(seg);   // ★ 逐段引用（剔掉共享角色块）

    const pics = (seg && seg.pictures) || [];
    for (const p of orderPicsForThumb(seg, pics, metas, perShot)) {
      const hit = pick(p);
      if (hit) return viewUrl(hit.file, "input");
    }
    // 兜底 A：逐段字段抠到的 <Picture N>
    for (const n of orderPicsForThumb(seg, perShot, metas, perShot)) {
      const hit = pick(n);
      if (hit) return viewUrl(hit.file, "input");
    }
    // 兜底 B：全字段（含共享角色块）
    const tags = segAllText(seg).match(/<Picture\s+(\d+)\s*>/gi) || [];
    const nums = [];
    for (const tag of tags) {
      const n = Number(String(tag).replace(/\D+/g, ""));
      if (n && nums.indexOf(n) < 0) nums.push(n);
    }
    for (const n of orderPicsForThumb(seg, nums, metas, perShot)) {
      const hit = pick(n);
      if (hit) return viewUrl(hit.file, "input");
    }
    // 抠不到就留空。★ 不再退到"本节点第一张已接图" —— 那是与段落无关的
    //   常量，会让所有段显示同一张（用户报的"全部都一样"就是这么来的）。
    return "";
  }

  /** 与 segRefMetas() 同口径，但用本闭包内可见的 refSlotImages/控件读取，
   *  避免跨作用域引用（这处 refThumbFor 与时间线那处不在同一闭包）。 */
  function localRefMetas() {
    try {
      if (state.classify) {
        const rows = state.classify();
        if (rows && rows.length) return rows;
      }
    } catch (e) { /* state 未就绪：继续读控件 */ }
    try {
      const w = findWidget(node, "ref_classify");
      if (w && w.value) return JSON.parse(w.value) || [];
    } catch (e) { /* 解析失败就当没有元数据 */ }
    return [];
  }

  function setPrompt(idx) {
    const segs = state.segments();
    const seg = idx > 0 ? segs[idx - 1] : null;
    who.replaceChildren();
    if (!seg) {
      who.appendChild(document.createTextNode("—"));
      hdPrompt.querySelector(".stat").textContent = "未选";
      bdPrompt.className = "h3d-plan-bd is-empty";
      bdPrompt.textContent = "还没解析 PACK，或者还没选中分镜。";
      // ★ 第 31 轮：把 editorEls 也清掉，配合下方存在分支的兜底守卫，
      //   双重保险 ——「plan 事件导致 setPrompt(0) → finish 切回 valid」这类
      //   中段回滚，不再卡在 placeholder 上。
      editorEls = null;
      return;
    }
    // shot_id 走中性徽标
    const ix = h("span", "h3d-ix", seg.id);
    who.appendChild(document.createTextNode("第 " + idx + " / " + segs.length + " 段 · "));
    who.appendChild(ix);

    // 分镜时间统计：取 PACK 段头 ########## S02 / 11s+1.6=12.6 / EN ##########
    // 里的时长标注（与段头同源）；取不到就按 new/handoff/gen 秒数拼等价文本。
    let tLabel = h3SegTimeLabel(state.packText(), seg.id, edLang);
    if (!tLabel) {
      const f = (v) => {
        const n = Number(v || 0);
        return Math.abs(n - Math.round(n)) < 1e-6 ? String(Math.round(n))
                                                  : String(Number(n.toFixed(2)));
      };
      const nS = Number(seg.new_seconds || 0);
      const hS = Number(seg.handoff_seconds || 0);
      const gS = Number(seg.gen_seconds || 0);
      tLabel = hS > 1e-6 ? f(nS) + "s+" + f(hS) + "=" + f(gS) : f(nS) + "s";
    }
    if (tLabel) {
      const tEl = h("span", "h3d-time", " · " + tLabel);
      tEl.title = Number(seg.handoff_seconds || 0) > 1e-6
        ? "分镜时间（取自 PACK 段头）：新增时长 + 段首回放 = 生成时长"
        : "分镜时间（取自 PACK 段头）：本段新增时长";
      who.appendChild(tEl);
    }

    // 整段六段式编辑区：EN/ZH 各一个 textarea，按当前 lang 显隐。
    // 正文优先**直接从 PACK 原文切**（不依赖后端 full_fields，改 JS 立即生效，
    // 且就是 PACK 原文）；切不到时退回 full_fields 拼接，再退回 seg.prompt。
    // ★ 第 31 轮：守卫不能只看 editorEls —— `!seg` 分支会把 bdPrompt 重置成
    //   placeholder，但忘了把 editorEls 也清掉；下次回到有效段时 buildEditor
    //   不跑，bdPrompt 一直卡在 placeholder。现象：点了「重渲该段」后端
    //   execute 一开始发 `event="plan"` → select(0) → setPrompt(0) 把 bdPrompt
    //   占了，再 finish 切回去 → setPrompt(valid) 走存在分支 → editorEls 非 null
    //   → bdPrompt 还是 placeholder，stat 却写出了有效段的词数。两种状态同时
    //   出现就是这条守卫没考虑过的"中段回滚"。
    //   改用「bdPrompt 当前是不是 placeholder 状态」当兜底触发：还在 placeholder
    //   就重跑一次 buildEditor，把 textarea 重新挂回去。
    if (!editorEls || bdPrompt.classList.contains("is-empty")) buildEditor();
    // 记下本次渲染的段号：切语言时要按它重填（见 applyLang）。
    loadedRef.current = idx;
    // ★ EN / ZH 两侧**都填**（不只当前 edLang 侧）。
    //   原来只填当前侧，切到从未填过的一侧就是空白 ——「中文 ZH 没内容」就是这么来的。
    //   现在一次把两版都填好，切 tab 立即有内容，也不依赖 tab 点击时再补。
    ["en", "zh"].forEach((lang) => {
      const ta = editorEls[lang];
      if (!ta) return;
      let block = h3ExtractSegBlock(state.packText(), seg.id, lang);
      let fromFallback = false;
      if (!block) {
        // 兜底：PACK 里切不到该语言版块时，用后端解析出的字段拼；
        // 英文侧再退一步用 seg.prompt（detailed_description）。
        const ff = lang === "en" ? (seg.full_fields || {}) : (seg.full_fields_zh || {});
        block = h3SegToBlock(ff, lang === "en" ? seg.prompt : "");
        // ★ 标记「这段正文不是 PACK 原文，是后端拼出来的」。doWrite 绝不把它
        //   写回 PACK —— 否则自动修正生成的模板会顶掉用户原文（实测清空过英文块）。
        fromFallback = !!block;
      }
      // 中文版切不出来时给明确提示：新版技能默认只交付英文版 PACK，没有中文源，
      // 别让用户对着空框猜是不是坏了。
      ta.placeholder = block
        ? (lang === "zh"
            ? "中文版六段式提示词（subject_definitions: … non_diegetic_music:）"
            : "英文版六段式提示词（subject_definitions: … non_diegetic_music:）")
        : (lang === "zh"
            ? "本 PACK 没有中文版分块（新版技能默认只交付英文版）—— 英文内容见「英文 EN」。"
            : "本段没有解析到提示词正文。");
      /* ★ 重填判定：只有当「文本框装的就是这一段这一语言」且「正在编辑它」时
         才跳过。换段/换语言一律强制重填 —— 否则残留内容会被写到错误的段上。
         同时记下身份与来源，供 doWrite 写入前校验。 */
      const sameTarget = editorLoaded.seg === idx && editorLoaded.lang === lang;
      const editing = sameTarget && ta === document.activeElement;
      if (!editing && ta.value !== block) ta.value = block;
      if (lang === edLang) {
        editorLoaded = { seg: idx, lang: lang, fromFallback: fromFallback };
      }
    });
    applyLang();

    const body = String(seg.prompt || "").trim();
    const words = body ? body.split(/\s+/).filter(Boolean).length : 0;
    const pics = (seg.pictures || []).length;
    /* ★ 第 30 轮：正文为空必须**看得见**。
       以前只是安静地写 "0 词 · 0 图"，和正常段长得一模一样 —— 用户以为
       是自己没选对段，反复点缩略图。实测踩过：PACK 在工作流 JSON 里存了
       两份（widgets_values / widgets_values_named），ComfyUI 加载了 S01
       英文块被清空的那一份，界面零提示。
       现在转红并说明原因；详情看「剧本 PACK」版块那条红字诊断。 */
    const stEl = hdPrompt.querySelector(".stat");
    if (words === 0) {
      stEl.textContent = "⚠ " + words + " 词 · " + pics + " 图 · 正文为空";
      stEl.className = "stat is-bad";
      stEl.title = "本段在 PACK 里没解析到正文，渲染出来会是一段默认运镜。\n"
        + "常见原因：\n"
        + "1) 工作流 JSON 里 PACK 存了两份（widgets_values /\n"
        + "   widgets_values_named），加载到了被清空的那一侧；\n"
        + "2) 这份 PACK 根本没有该语言版块（如只有中文版）。\n"
        + "对照上方「剧本 PACK」版块的红字诊断。";
    } else {
      stEl.textContent = words + " 词 · " + pics + " 图";
      stEl.className = "stat";
      stEl.removeAttribute("title");
    }
  }

  function setPreview(idx) {
    if (!cardPrev) return;      // parser：没挂预览栏，直接跳过
    const segs = state.segments();
    const seg = idx > 0 ? segs[idx - 1] : null;
    whoPrev.replaceChildren();
    bdPrev.replaceChildren();
    if (!seg) {
      whoPrev.appendChild(document.createTextNode("—"));
      hdPrev.querySelector(".stat").textContent = "未选";
      bdPrev.className = "h3d-plan-bd is-empty";
      bdPrev.textContent = "选中分镜后这里会显示对应视频 / 首帧。";
      return;
    }
    const ix = h("span", "h3d-ix", seg.id);
    whoPrev.appendChild(document.createTextNode("第 " + idx + " / " + segs.length + " 段 · "));
    whoPrev.appendChild(ix);

    const url = state.renderedUrl(idx);
    const nS = Number(seg.new_seconds || 0);

    /* ★ 整卡遮罩（.pveil）：与参考图卡片同一套交互语言 —— 图库类产品做法。
       这是**只读媒体**，铺遮罩不会挡住原生交互（左栏 textarea 就不铺）。
       结构同构于 .h3d-ref .veil：.ptop 顶部细行 / .pkind 中间大字 / .pacts 底部操作。
       ★ 遮罩层 pointer-events:none，只有按钮复活 pointer-events:auto
         —— 否则 hover 时 video 的 controls 会被整层吃掉，拖不了进度。 */
    function buildVeil() {
      const veil = h("div", "pveil");
      const top = h("div", "ptop");
      top.appendChild(h("span", "pst " + (url ? "is-ok" : "is-bad"),
        url ? "已渲染" : "未渲染"));
      top.appendChild(h("span", "psid", "第 " + idx + " / " + segs.length + " 段"));
      const kind = h("div", "pkind");
      kind.appendChild(document.createTextNode(seg.id || ("S" + idx)));
      kind.appendChild(h("span", "sub", nS.toFixed(1) + "s · 新增时长"));
      const acts = h("div", "pacts");
      if (url) {
        const bPlay = h("button", "pact is-play", "⤢ 全屏");
        bPlay.title = "在新窗口打开本段完整视频（同段块双击）";
        bPlay.onclick = (e) => { e.stopPropagation(); window.open(url, "_blank"); };
        const bRe = h("button", "pact", "↻ 重渲");
        bRe.title = "只重渲第 " + idx + " 段（首尾锚定，前段复用缓存）";
        bRe.onclick = (e) => { e.stopPropagation(); repairSegment(idx); };
        acts.appendChild(bPlay);
        acts.appendChild(bRe);
      } else {
        const bRe = h("button", "pact is-play", "↻ 渲染本段");
        bRe.title = "只渲染第 " + idx + " 段（首尾锚定，前段复用缓存）";
        bRe.onclick = (e) => { e.stopPropagation(); repairSegment(idx); };
        acts.appendChild(bRe);
      }
      veil.appendChild(top);
      veil.appendChild(kind);
      veil.appendChild(acts);
      return veil;
    }

    if (url) {
      // 已渲染：默认静默首帧 + 不显示原生控件（避免截图里那种「一上来就
      // 出现播放/进度条/音量/全屏/三点菜单」的全套播放器 UI）。
      // 用户要求"鼠标滑过是点击播放视频的"——所以默认无控件、hover 视频
      // 才显示 controls 并自动播放；离开即暂停 + 回首帧 + 控件收回。
      // ★ muted=true 是 Chrome/Safari 自动播放策略硬要求（即使 mouseenter
      //   也算 user gesture，但「跨元素连续触发 + 同时显示控件」路径上
      //   多家浏览器仍然要求 muted；非 muted 会被拒掉并打印 not-allowed）。
      const v = document.createElement("video");
      v.src = url;
      v.muted = true;
      v.loop = true;
      v.playsInline = true;
      v.controls = false;
      v.preload = "metadata";
      // 强制首帧：设完 src 后把 currentTime 钉到 0；并禁用 autoplay。
      v.addEventListener("loadedmetadata", () => {
        try { v.currentTime = 0; } catch (e) {}
      }, { once: true });
      // hover 显示控件 + 自动播放；离开即停。每次 setPreview 创建新 <video>，
      // 旧元素随 bdPrev.replaceChildren() 被丢弃，listener 自动释放，无累积。
      v.addEventListener("mouseenter", () => {
        v.controls = true;
        try { v.play().catch(() => {}); } catch (e) {}
      });
      v.addEventListener("mouseleave", () => {
        try { v.pause(); } catch (e) {}
        try { v.currentTime = 0; } catch (e) {}
        v.controls = false;
      });
      bdPrev.className = "h3d-plan-bd is-preview";
      bdPrev.appendChild(v);
      // ★ 已渲染分支不再铺 buildVeil() 玻璃浮层 —— 它会把视频画面
      //   "罩"在 backdrop-filter 模糊层下面，hover 时看上去一团糊，
      //   跟"鼠标滑过能看到视频"的诉求冲突。重渲入口走顶部"重渲已改段"，
      //   单段重渲也可以点时间线段块右键菜单（已存在），不需要在视频上
      //   浮按钮。未渲染分支继续铺，那里没视频、遮罩合理。
      hdPrev.querySelector(".stat").textContent = "已渲 · 悬停预览";
      hdPrev.querySelector(".stat").className = "stat is-ok";
      cardPrev.classList.remove("is-bad");
    } else {
      // 未渲染：本段首张参考图 + "未渲染" 提示，同样铺遮罩
      bdPrev.className = "h3d-plan-bd is-preview";
      const ph = h("div", "ph");
      const row = h("div", "ph-row");
      const phImg = document.createElement("img");
      const phSrc = refThumbFor(seg);
      if (phSrc) phImg.src = phSrc;
      const ic = h("div", "lb", "▶");
      ic.style.cssText = "font-size:18px;color:#4fff8f;opacity:.6;";
      const lb = h("div", "lb", "未渲染 · 显示首帧参考图");
      row.appendChild(ic);
      row.appendChild(lb);
      ph.appendChild(row);
      if (phSrc) ph.appendChild(phImg);
      bdPrev.appendChild(ph);
      bdPrev.appendChild(buildVeil());
      hdPrev.querySelector(".stat").textContent = "未渲染";
      hdPrev.querySelector(".stat").className = "stat is-bad";
    }
  }

  // 选中状态视觉：高亮左/右卡片 + 时间线缩略图
  function applySelection() {
    const idx = state.planSelectedIndex;
    if (idx > 0) {
      cardPrompt.classList.add("is-selected");
      if (cardPrev) cardPrev.classList.add("is-selected");
    } else {
      cardPrompt.classList.remove("is-selected");
      if (cardPrev) cardPrev.classList.remove("is-selected");
    }
  }

  // parser 段按钮：点哪个就切到哪一段的完整提示词
  function refreshSegBtns() {
    if (!segBtns) return;
    const segs = state.segments() || [];
    segBtns.replaceChildren();
    segs.forEach((s, i) => {
      const idx = i + 1;
      const b = h("button",
        "h3d-lang-tab" + (idx === state.planSelectedIndex ? " is-active" : ""), s.id);
      b.title = "第 " + idx + "/" + segs.length + " 段 · 新增 "
        + Number(s.new_seconds || 0).toFixed(1) + "s";
      b.onclick = () => select(idx);
      segBtns.appendChild(b);
    });
  }

  function render() {
    let idx = state.planSelectedIndex || 0;
    // idx = 0 表示"未选"，>0 是真正选中的段号。
    // parser 上没有时间线跟随，PACK 一解析出来就默认展示第 1 段，
    // 免得「完整分镜提示词」区一直空着等用户点（Director 保持未选语义不变）。
    if (idx === 0 && parserMode && (state.segments() || []).length > 0) {
      idx = 1;
      state.planSelectedIndex = 1;
    }
    refreshSegBtns();
    setPrompt(idx);
    setPreview(idx);
    applySelection();
    // ★ 内容一变就得重贴节点高度，否则 legacy 下 DOM widget 容器会停在
    //   上一次量出来的旧高度上 —— 实测差 16 CSS px（屏幕上 ~14px），
    //   面板底部会露出一截没被节点背景包住。
    //   跟 buildPackPanel 解析完调 refit(node) 是同一个道理。
    refit(node);
  }

  // 暴露给 mountPanels：被 timeline 缩略图 click 调用
  function select(idx) {
    state.planSelectedIndex = idx > 0 ? idx : 0;
    render();
  }

  // 监听 Director 进度推送：渲染中的段自动成为"选中"
  function planProgressHandler(ev) {
    const d = ev.detail || {};
    if (String(d.node_id) !== String(node.id)) return;
    if (d.event === "segment_start") {
      // 用户没手动选过时（-1 / 0），跟随 activeIndex；已手动选过则不抢
      if (!state.planSelectedIndex || state.planSelectedIndex < 0) {
        select(d.index);
      }
    } else if (d.event === "finish") {
      // 全部完成：选最后一段
      const last = state.segments().length;
      if (last > 0) select(last);
    } else if (d.event === "plan") {
      // 重新规划时清掉旧选择（state.planSelectedIndex 保持 0）
      select(0);
    }
  }
  api.addEventListener("h3.director.progress", planProgressHandler);

  // 重选按钮：回到空闲
  btnReset.onclick = (e) => { e.stopPropagation(); select(0); };

  // 初次渲染
  render();

  // 只返回内容块；由 mountPanels 挂进时间线的 bd。
  return { body, render, select,
           cleanup: () => {
             try { api.removeEventListener("h3.director.progress", planProgressHandler); }
             catch (e) { /* 不支持移除就算了 */ }
           } };
}

/* ------------------------------------------------------------------ */
/* 装配                                                                */
/* ------------------------------------------------------------------ */
// app 拿不到时不要让它把整个模块炸掉：换成空实现，至少别的节点不受影响，
// 控制台也能看到明确原因。
/* ------------------------------------------------------------------ */
/* 挂载时机：自定义节点 JS 是异步 import 的，等它跑起来时画布节点早已建好，
 *  nodeCreated 不会补触发已存在的节点。故三条路全挂（init/afterConfigureGraph
 *  + 轮询兜底 + nodeCreated），靠 node.__h3dMounted 做幂等。             */
/* ------------------------------------------------------------------ */

/** 节点类型：1.51.10 内部一律读 constructor.comfyClass，这里三种都试。 */
function nodeClass(node) {
  try {
    return ((node && node.constructor && node.constructor.comfyClass)
            || (node && node.comfyClass) || (node && node.type)) || "";
  } catch (e) {
    return "";
  }
}

// 面板挂载目标：H3Director 挂完整面板；H3PromptPackParser 只挂「剧本 PACK +
// 分镜提示词编辑器」（parser 没有 run_segments / 进度事件这些 Director 专属控件）。
// 原来两套编辑器（h3_shot_editor.js 挂 parser+director、h3_director.js 挂 director）
// 在 700 上叠成两个 DOM widget 面板互相打架 —— 现在统一由本文件承担。
function isDirector(node) {
  const c = nodeClass(node);
  return c === "H3Director" || c === "H3PromptPackParser";
}

/** 当前节点是不是 H3PromptPackParser（决定挂完整面板还是精简面板）。 */
function isParserNode(node) {
  return nodeClass(node) === "H3PromptPackParser";
}

/**
 * 单独构建一个子面板。崩了就返回一个"这个模块挂了"的占位块 + 空方法，
 * 保证另外两个子面板和整个 DOM widget 都还能出来。
 */
function safePanel(name, fn) {
  try {
    return fn();
  } catch (e) {
    const msg = (e && e.message) || String(e);
    diag.lastErr = name + " 面板: " + msg;
    console.error("[H3 Director] 子面板[" + name + "] 构建失败：", e);
    const sec = h("div", "h3d-sec");
    const hd = h("div", "h3d-hd");
    const badge = h("span", "h3d-ix", "!");
    hd.appendChild(badge);
    hd.appendChild(h("span", "h3d-title", "⚠ " + name));
    hd.appendChild(h("span", "h3d-tag is-bad", "构建失败"));
    const bd = h("div", "h3d-bd");
    bd.appendChild(h("div", "h3d-status is-error", msg));
    sec.appendChild(hd);
    sec.appendChild(bd);
    return { sec, badge, parse() {}, render() {}, refreshRendered() {} };
  }
}

/* ---- 诊断条：不开 devtools 也能一眼看到扩展跑到哪一步了 ---- */
const diag = { appOk: !!(app && app.registerExtension), graphOk: false,
               found: 0, mounted: 0, lastErr: "", el: null };

function diagRender() {
  const el = diag.el;
  if (!el || !el.isConnected) return;
  const lines = [
    "app 可用: " + ((app && app.registerExtension) ? "是" : "否  (还没注册上)"),
    "画布已就绪: " + (diag.graphOk ? "是" : "否"),
    "图上 H3Director: " + diag.found + " 个",
    "已挂上面板: " + diag.mounted + " 个",
  ];
  if (diag.lastErr) lines.push("最近错误: " + diag.lastErr);
  el.textContent = "";
  // 挂上了就别一直杵在屏幕上：6 秒后自己消失。挂不上才常驻等你来看。
  if (diag.mounted > 0 && !diag.lastErr && !diag._armed) {
    diag._armed = true;
    setTimeout(() => { if (diag.el) { diag.el.remove(); diag.el = null; } }, 6000);
  }
  const title = h("div");
  title.textContent = "ComfyUI-Marquee-Director 诊断";
  title.style.cssText = "font-weight:700;margin-bottom:4px;";
  el.appendChild(title);
  lines.forEach((s) => { const d = h("div"); d.textContent = s; el.appendChild(d); });
}

function diagInit() {
  if (diag.el || typeof document === "undefined" || !document.body) return;
  const el = h("div");
  el.id = "h3d-diag";
  el.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:99999;"
    + "min-width:200px;max-width:300px;padding:8px 12px 10px;"
    + "border:1px solid #4a9;border-radius:8px;"
    + "background:rgba(8,20,18,.94);color:#bfe;font:11px/1.7 system-ui,sans-serif;"
    + "white-space:pre-wrap;word-break:break-all;box-shadow:0 4px 18px rgba(0,0,0,.5);";
  const x = h("div");
  x.textContent = "×";
  x.style.cssText = "position:absolute;top:1px;right:7px;cursor:pointer;"
    + "opacity:.65;font-size:15px;line-height:1;";
  x.onclick = () => { el.remove(); diag.el = null; };
  el.appendChild(x);
  diag.el = el;
  document.body.appendChild(el);
  diagRender();
}

/** 扫一遍画布，给所有还没挂面板的 H3Director 挂上。 */
function sweep() {
  try {
    if (!app) app = _fromWindow("app");
    const g = app && app.graph;
    if (!g) return 0;
    diag.graphOk = true;
    const list = g._nodes || g.nodes || [];
    let found = 0;
    for (const n of list) {
      if (!isDirector(n)) continue;
      found += 1;
      mountPanels(n);
    }
    diag.found = found;
  } catch (e) {
    diag.lastErr = "sweep: " + ((e && e.message) || e);
  }
  diagRender();
  return diag.found;
}

/** 把面板装到一个 H3Director 节点上。幂等：装过就不重复装。 */
function mountPanels(node) {
    if (!node || node.__h3dMounted) return;
    try {
    injectCss();

    const state = {
      packData: null,
      classifyRows: [],
      rendered: new Map(),
      // 磁盘上「从 1 开始连续渲完」的段数，来自 /h3/session 的 done 字段。
      // 断点渲染前先问一次，好把"从第几段继续"写在提示里。
      done: 0,
      // 当前「功能规划」版块选中的段号（0 = 未选；1..N 才是真段号）。
      // 默认跟随正在渲染的分镜；用户点击分镜缩略图会改写它。
      planSelectedIndex: 0,
      // 「提示词已改、待重渲」的段号集合（1-based）。编辑提示词写回 PACK 后加进来；
      // 时间线据此打「✎ 已改」角标并提供一键重渲；该段真的开始重渲时清掉。
      dirtySegs: new Set(),
      // 提示词写回模式：
      //   "pack"     —— 写回外侧 PACK 文本框（PrimitiveStringMultiline，默认）
      //   "override" —— 写进 H3PromptPackParser 的 pack_override（不动原文，
      //                 execute 里优先读它）。原来是 h3_shot_editor.js 的独立开关，
      //                 现并入本文件统一管理。
      writeMode: "pack",
      // 当前 PACK 源文本：统一走模块级 h3PackText()，与编辑器 / 写回共用同一份实现
      // （原来这里内联了一份一模一样的查找逻辑，两处容易走偏）。
      packText: () => h3PackText(),
      classify: () => {
        if (state.classifyRows.length) return state.classifyRows;
        const w = findWidget(node, "ref_classify");
        try {
          return JSON.parse(w && w.value ? w.value : "[]") || [];
        } catch (e) {
          return [];
        }
      },
      onPack: (d) => {
        state.packData = d;
        // 子面板可能构建失败（那时它是个空壳）、parser 上部分面板压根不挂（null），
        // 所以每个都要判空，别让回调把整条链打断。
        try { output && output.render && output.render(); } catch (e) { /* 已降级/未挂 */ }
        try { timeline && timeline.render && timeline.render(); } catch (e) { /* 已降级/未挂 */ }
        try { refs && refs.render && refs.render(); } catch (e) { /* 已降级/未挂 */ }
        // 功能规划版块的「分镜提示词」编辑区要从最新解析结果刷新内容
        // （尤其 PACK 被外部/写回接口改过之后），否则显示会陈旧。
        try { plan.render && plan.render(); } catch (e) { /* 面板已降级 */ }
        // PACK 解析完才知道会话名（session_name 控件空着时用它），所以
        // 「已渲染到第几段」必须等这一下再拉 —— 首屏那次 refreshRendered
        // 是在 pack 解析之前跑的，会话名还是 my_chain，拉回来是空的。
        try { timeline && timeline.refreshRendered && timeline.refreshRendered(); }
        catch (e) { /* 已降级/未挂 */ }
      },
      onClassify: (rows) => {
        state.classifyRows = rows;
      },
      segments: () => (state.packData ? state.packData.segments : []),
      renderedUrl: (i) => state.rendered.get(i) || "",
    };

    // 节点删除时统一清理监听器 / 观察器，避免内存泄漏和删节点后报错。
    const cleanups = [];
    function addCleanup(fn) { cleanups.push(fn); }

    const root = h("div", "h3d");
    // ★ 自愈历史脏值：老工作流里存过被撑爆的高度（实测 2,693,023px），加载
    //   瞬间画布就卡住，等不及第一次 panelLayout 把它压回来。挂载前先收回安全
    //   范围；正常工作流（高度本来就 < 上限）不受影响。
    try {
      if (node && node.size && node.size[1] > H3D_PANEL_MAX_H * 2) {
        node.size[1] = H3D_PANEL_MAX_H;
      }
    } catch (e) { /* 忽略 */ }
    // 版块顺序 = 人的操作顺序：先填参数 → 定怎么切 → 读剧本 →
    // 定成片规格 → 调画质 → 输出开关 → 核参考图 → 看渲染进度 → 核当前规划。
    // 每个子面板各自独立构建：任何一个崩掉都只坏它自己，其余照常显示。
    // （以前是一整块 try，时间线里一个 SVG 画不出来就整个面板都没了。）
    // parser 节点（720）：没有 run_segments / 进度事件等 Director 专属控件，
    // 只挂「剧本 PACK + 分镜提示词编辑器」，参数 / 输出规格 / 参考图 / 时间线全跳过。
    const parserOnly = isParserNode(node);
    const built = [];
    if (!parserOnly) {
      PARAM_SECTIONS.forEach((spec) => {
        built.push(safePanel(spec.title, () => buildWidgetSection(node, spec)));
      });
    }
    // 「剧本 PACK」概览版块：Director 上正常显示；parser 上**不显示**（头部信息
    // 项目/模式/时长/段数/画幅与「完整分镜提示词」区功能重复），但仍要构建它
    // 当「解析触发器」——它的 parse() 负责 POST /h3/pack_preview 并把结果推给
    // plan.render()，去掉会断掉「改 PACK → 编辑器刷新」这条链。
    const pack = safePanel("剧本 PACK", () => buildPackPanel(node, state));
    const output = parserOnly ? null
      : safePanel("输出规格", () => buildOutputPanel(node, state));
    const refs = parserOnly ? null
      : safePanel("参考图", () => buildRefPanel(node, state));
    const timeline = parserOnly ? null
      : safePanel("分镜时间线", () => buildTimelinePanel(node, state));
    const plan = safePanel("功能规划", () => buildPlanPanel(node, state));
    // 子面板注册的全局监听器在节点删除时统一清理。
    if (timeline && timeline.cleanup) addCleanup(timeline.cleanup);
    if (plan && plan.cleanup) addCleanup(plan.cleanup);
    // ★ Director：「功能规划」挂进「分镜时间线」的 bd，受时间线标题折叠控制，
    //   所以不进 built（不占序号、不另起 .h3d-sec）。
    // ★ parser：没有时间线版块，给编辑器自己起一个独立 sec 并进 built 拿编号。
    if (timeline && timeline.bd && plan && plan.body) {
      timeline.bd.appendChild(plan.body);
    } else if (plan && plan.body) {
      const sec = h("div", "h3d-sec");
      const hd = h("div", "h3d-hd");
      hd.appendChild(h("span", "h3d-title", "分镜提示词编辑器"));
      sec.appendChild(hd);
      const bd = h("div", "h3d-bd");
      bd.appendChild(plan.body);
      sec.appendChild(bd);
      plan.sec = sec;          // 让下面的编号逻辑认到这个版块
      built.push(plan);
    }
    // 序号统一在这里发：某个版块因为"这个节点上没这几个参数"被跳过时，
    // 后面的编号会自动顶上来，不会出现 1 / 3 / 4 这种断号。
    // parser 上不把 pack 加进 built（不显示、不占编号），只当解析触发器用
    if (!parserOnly) built.push(pack);
    built.push(output, refs, timeline);
    // 「功能规划」选中态要在时间线缩略图上同步高亮——
    // 时间线 click 也得回头通知 plan，两边都用 state.planSelectedIndex 当真相源。
    // 切段时不要 timeline.render() 整段重建（会重置所有 hover 弹层和 video），
    // 只走 timeline.setPlanHighlight() 翻一翻 className。
    state.planSelect = (idx) => {
      state.planSelectedIndex = idx > 0 ? idx : 0;
      plan.render();
      if (timeline && timeline.setPlanHighlight) timeline.setPlanHighlight();
    };
    // 提示词被编辑后由「功能规划」左栏回调：记下待重渲的段号，并让时间线打角标。
    // 用回调而不是让 buildPlanPanel 直接引用 timeline —— 两个子面板互不认识，
    // 统一走 state 这个共享真相源（和 planSelect 一个套路）。
    state.onDirty = (idx) => {
      if (!idx || idx <= 0) return;
      state.dirtySegs.add(idx);
      if (timeline && timeline.refreshDirty) timeline.refreshDirty();
    };

    /* ---- 参考图变化「统一监听」：换图 / 断线 / 新接 → 所有用图的地方一起刷 ----
     *
     * 需求：「加载图片或换图后，参考图面板 + 分镜时间线缩略图 + 参与渲染的
     *        参考图全部自动刷新，逻辑连贯」。
     *
     * ★★ 为什么放在 mountPanels 这一层，而不是各面板内部：
     *   分镜时间线的缩略图（refThumbFor）和参考图卡片**吃的是同一个
     *   refSlotImages(node)**。如果监听只挂在参考图面板里，就会出现
     *   「卡片换成新图了、时间线缩略图还是旧的」这种半新半旧 —— 反而更迷惑。
     *   所以在这里统一监听，一处变化、通知全部视图。
     *
     * 双保险（两条路缺一不可）：
     *   ① onConnectionsChange —— 连线变动（新接 / 断开），即时。
     *   ② 轮询签名 —— widget 值变动。★ 换图**往往不动线**，只改 LoadImage 的
     *      image 控件值，① 完全抓不到。改动途径又特别多（画布下拉、拖入上传、
     *      API、Ctrl+Z 撤销），逐个给上游 widget 挂 callback 会在节点重渲染时
     *      丢失 —— 轮询签名是唯一覆盖全部途径的稳妥办法。
     *
     * 签名 = 各槽位「槽号:文件名」拼串；只在签名真的变了才重绘，避免每秒白刷。
     */
    function refSigNow() {
      try {
        return refSlotImages(node).map((s) => s.slot + ":" + (s.file || "")).join("|");
      } catch (e) {
        return "";
      }
    }
    let _refSig = refSigNow();
    let _refWatch = null;

    /** 参考图相关的所有视图一起重绘（顺序：先轻后重） */
    function renderAllRefViews() {
      try { refreshTokenViews(); } catch (e) {}                    // 提示词胶囊
      try { if (refs && refs.render) refs.render(); } catch (e) {}  // 参考图卡片
      try { if (timeline && timeline.render) timeline.render(); } catch (e) {}  // 时间线缩略图
    }
    function stopRefWatch() {
      if (_refWatch) { clearInterval(_refWatch); _refWatch = null; }
    }

    if (!parserOnly) {
      _refWatch = setInterval(() => {
        // 面板已脱离文档（节点删了 / 折叠重建）→ 停表，别越积越多越跑越慢
        if (!root.isConnected) { stopRefWatch(); return; }
        let sig = "";
        try { sig = refSigNow(); } catch (e) { return; }
        if (sig === _refSig) return;
        _refSig = sig;
        renderAllRefViews();
      }, 1000);
      addCleanup(stopRefWatch);        // 节点删除时统一停表

      // 连线变动：不等下一次轮询，立刻全量刷新
      try {
        const _prevConn = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
          try { if (_prevConn) _prevConn.apply(this, args); } catch (e) {}
          try {
            _refSig = refSigNow();     // 先同步签名，免得紧接着又重绘一次
            renderAllRefViews();
          } catch (e) {}
        };
      } catch (e) { /* 老版本 litegraph 没有这个钩子 */ }
    }
    state.clearDirty = (idx) => {
      if (idx && idx > 0) state.dirtySegs.delete(idx);
      else state.dirtySegs.clear();
      if (timeline && timeline.refreshDirty) timeline.refreshDirty();
    };
    let no = 0;
    built.forEach((p) => {
      if (!p || !p.sec) return;
      no += 1;
      try { if (p.badge) p.badge.textContent = String(no); } catch (e) { /* 降级壳没徽标 */ }
      root.appendChild(p.sec);
    });
    // 参数面板 = built 里前 PARAM_SECTIONS.length 个（可能有个别为 null）。
    // parser 上压根没挂参数版块，直接空数组，别把 pack/plan 误当参数面板重复 render。
    const paramPanels = parserOnly ? []
      : built.slice(0, PARAM_SECTIONS.length).filter((p) => p && p.sec);
    // 纯数据字段：不用露出来，但也不能让它们把面板顶到下面去
    SILENT_WIDGETS.forEach((nm) => hideWidget(node, nm));
    // parser 上额外藏掉 pack_override —— 它是编辑器内部用的覆盖通道，
    // 露出来只会让用户误改（详见 pack_nodes.py 的 schema 注释）。
    if (parserOnly) hideWidget(node, "pack_override");
    // ★ Nodes 2.0 下 Vue 会在自己重渲染时（比如触发一次 props/state 更新）
    //   把我们手动打的隐藏样式覆盖回去，被代理的原生控件（"全局提示词"
    //   "分镜提示词"这些）就会复活，跟面板里的代理内容重复。只在挂载时
    //   隐藏一次治不了这个——只能定时重钉。500ms 够快，肉眼看不出"先露出
    //   来一下又被藏起来"的闪烁；hideWidget 内部全是幂等操作，没有额外副作用。
    try {
      const hiddenTimer = setInterval(() => {
        ALL_HIDDEN_WIDGET_NAMES.forEach((nm) => hideWidget(node, nm));
        /* ★ 容器尺寸也要一起重钉 —— 「显示/隐藏高级输入」两套状态空白不
         *   一样就是漏了这条：
         *   切换 advanced 会改变节点上原生控件的数量 → Vue 重渲染 → 我们在
         *   panelLayout 里打在容器上的 height:auto 被覆盖回 size-full 的
         *   100% → 容器高度不再由内容决定，而是「吃掉 grid 行剩余空间」。
         *   于是隐藏高级输入时原生控件变少、容器分到的空间反而变大 → 底部
         *   凭空多出一截空白；显示时控件多、容器分到的空间刚好≈内容高度，
         *   看起来就正常。两个状态一对比，空白明显不一样。
         *   和 hideWidget 同理：一次性补丁扛不住框架重渲染，必须持续重钉。
         *   __h3dRelayout 就是 panelLayout，内部会重新写 height:auto。 */
        try { if (node.__h3dRelayout) node.__h3dRelayout(); } catch (e) { /* 还没挂上 */ }
      }, 500);
      addCleanup(() => { try { clearInterval(hiddenTimer); } catch (e) { /* 忽略 */ } });
    } catch (e) { /* 不支持定时器就算了 */ }

    const dom = node.addDOMWidget("h3_director_panels", "div", root, {
      serialize: false,
      hideOnZoom: false,
      getValue() { return ""; },
      setValue() {},
    });
    // ★ 尺寸必须**跟着节点走**，不能自己定死：
    //   · 宽度用 ComfyUI 传进来的 width（就是节点内容宽），不要再自己减边距 ——
    //     减了面板就比节点窄一圈，看着像"浮在节点上的独立面板"；
    //   · 高度用内容真实高度，节点才会随内容长高/收窄（折叠版块时也会跟着缩）。
    //   root 没有设 height，offsetHeight 就是内容高度，不会和容器互相套娃。
    //   panelLayout 同时喂 legacy(computeSize) 和 Nodes 2.0(computeLayoutSize)
    //   两条路 —— 只喂一条的话，切换 Nodes 2.0 开关时高度就会算错。
    panelLayout(dom, root, node);
    node.__h3dRelayout = () => {
      try { panelLayout(dom, root, node); } catch (e) { /* 还没挂上 */ }
      try { node.setDirtyCanvas(true, true); } catch (e) { /* 老版本没有 */ }
    };
    /* 内容尺寸变了（版块展开/折叠、时间线换行、参考图增删、拖动节点边缘）
     * 都要跟上。★ 这里必须走 refit（重设节点高度），不能只刷 panelLayout ——
     *   只刷布局的话节点高度还停在旧值，面板照样被裁。
     * ★ 原来给 refit 加过"30ms 合流 + 40 次上限 + 3 秒静置重置"的节流，
     *   代价是：连续拖动节点边缘时每次 mousemove 都会触发一次观察，30ms
     *   合流下 40 次上限不到 1.5 秒就打满，之后 refit 被直接吞掉、不再
     *   执行 —— 拖拽期间背景/尺寸卡住不跟，松手后也不会再补一次，正是
     *   「背景不跟随」的另一个源头。改成纯 30ms 尾随合流、不设调用次数
     *   上限；refit() 内部本来就有"高度没变就不 setSize"的早退判断，
     *   去掉次数上限也不会跑出死循环。
     * ★ 同时观察 root（内容自身高度变化）和 wrap/.dom-widget 容器（节点
     *   整体宽度变化会反映到它身上）——legacy 下 node.onResize 会触发，
     *   但 Nodes 2.0/Vue 节点走的是另一套布局路径，不一定会调用底层
     *   LGraphNode.onResize，这层 ResizeObserver 是兜底，两条腿走路。 */
    try {
      if (typeof ResizeObserver !== "undefined") {
        let roTimer = 0;
        const scheduleRefit = () => {
          if (roTimer) return;
          roTimer = setTimeout(() => {
            roTimer = 0;
            try { if (node.__h3dRelayout) node.__h3dRelayout(); } catch (e) { /* 忽略 */ }
            try { refit(node); } catch (e) { /* 忽略 */ }
          }, 30);
        };
        const ro = new ResizeObserver(scheduleRefit);
        ro.observe(root);
        const wrapEl = root.parentElement;
        if (wrapEl && wrapEl !== root) ro.observe(wrapEl);
        addCleanup(() => { try { ro.disconnect(); } catch (e) { /* 忽略 */ } });
      }
    } catch (e) { /* 不支持就算了 */ }

    if (node.size[0] < 420) node.setSize([Math.max(480, node.size[0]), node.size[1]]);
    // 节点宽度刚被调整过，必须让面板布局立即跟上，否则首屏会出现紫底比面板宽。
    try { if (node.__h3dRelayout) node.__h3dRelayout(); } catch (e) { /* 忽略 */ }
    // 用户拖动节点边缘改变大小时，computeSize 不一定实时刷宽度，这里补一个钩子。
    try {
      const origResize = node.onResize;
      node.onResize = function(...args) {
        // 先同步宽度锚点（min-width），再 refit：拖宽/拖窄都可能因为
        // 参考图网格 auto-fill 换行而改变内容高度，只刷宽度不刷高度的话
        // 高度要等 ResizeObserver 30ms 之后才补上，肉眼可见地"背景先宽
        // 后高，跟手感差一拍"。
        try { if (node.__h3dRelayout) node.__h3dRelayout(); } catch (e) { /* 忽略 */ }
        try { refit(node); } catch (e) { /* 忽略 */ }
        if (typeof origResize === "function") return origResize.apply(this, args);
      };
    } catch (e) { /* 忽略 */ }
    // 对"早就创建好"的节点，光加 widget 不会触发重排，得手动要求重画一次，
    // 否则面板挂在节点上了但节点没长高，看起来还是没有。
    try { node.setDirtyCanvas(true, true); } catch (e) { /* 老版本没有这方法 */ }

    // 首屏解析 + 监听上游 PACK 文本框（边改边刷新）
    setTimeout(() => {
      try {
        paramPanels.forEach((p) => p.render && p.render());
      } catch (e) { console.error("[H3 Director] 节点参数首屏失败：", e); }
      try { pack.parse && pack.parse(); } catch (e) { console.error("[H3 Director] PACK 首屏解析失败：", e); }
      try { output && output.render && output.render(); } catch (e) { console.error("[H3 Director] 输出规格首屏失败：", e); }
      try { refs && refs.render && refs.render(); } catch (e) { console.error("[H3 Director] 参考图首屏渲染失败：", e); }
      try { timeline && timeline.refreshRendered && timeline.refreshRendered(); } catch (e) { console.error("[H3 Director] 时间线首屏失败：", e); }
      try {
        // 监听上游 PACK 文本框（边改边刷新）—— parser 查找统一走模块级 h3FindParser()
        const parser = h3FindParser();
        const src = parser ? upstream(parser, "pack_text") : null;
        if (src) {
          (src.widgets || []).forEach((w) => {
            if (w && w.element && w.element.querySelector) {
              const ta = w.element.querySelector("textarea");
              if (ta) ta.addEventListener("input", debounce(() => pack.parse && pack.parse(), 1200));
            }
          });
        }
      } catch (e) { /* 找不到就只知道手动点解析 */ }
    }, 400);

    // 节点被从画布上删除时，把本节点注册的全局监听器、观察器全部摘掉。
    try {
      const origRemoved = node.onRemoved;
      node.onRemoved = function(...args) {
        cleanups.forEach((fn) => { try { fn(); } catch (e) { /* 清理失败不影响删除 */ } });
        if (typeof origRemoved === "function") return origRemoved.apply(this, args);
      };
    } catch (e) { /* 老版本没有 onRemoved */ }

    node.__h3dMounted = true;
    diag.mounted += 1;
    } catch (err) {
      node.__h3dMounted = true;   // 失败也标记，免得每轮 sweep 重复报错刷屏
      diag.lastErr = (err && err.message) || String(err);
      showPanelError(node, err);
    }
    diagRender();
}

/* 注册是可以重试的：模块求值时 app 可能还没挂上，这时候不能就这么算了，
 * 得让轮询每轮再试一次，直到注册成功。 */
let _registered = false;

function registerNow() {
  if (_registered) return true;
  if (!app) app = _fromWindow("app");
  if (!app || typeof app.registerExtension !== "function") return false;
  try {
    app.registerExtension({
      name: EXT,

      // 画布恢复/初始化完成后立刻补扫（这是重启后能看到面板的关键）
      init() { diagInit(); queueSweep(); },
      setup() { diagInit(); queueSweep(); },
      afterConfigureGraph() { setTimeout(sweep, 300); },

      // 这条路只对之后"新建/新拖进来"的节点有效
      nodeCreated(node) {
        if (isDirector(node)) mountPanels(node);
      },
    });
    _registered = true;
    return true;
  } catch (e) {
    diag.lastErr = "registerExtension: " + ((e && e.message) || e);
    return false;
  }
}

/** 注册晚于画布恢复时，靠这几拍把已经存在的节点补上。 */
function queueSweep() {
  setTimeout(sweep, 0);
  setTimeout(sweep, 800);
  setTimeout(sweep, 2500);
}

if (!registerNow()) {
  console.warn("[H3 Director] app 暂不可用，扩展还没注册上；轮询会持续重试。");
}

/* 兜底轮询：注册重试 + 节点补扫。万一上面几个 hook 都没轮到，这里也能扫到。 */
let _ticks = 0;
setInterval(() => {
  _ticks += 1;
  if (_ticks > 180) return;           // 三分钟后停，别一直占着
  if (!_registered) { if (registerNow()) queueSweep(); }
  sweep();
}, 1000);

if (typeof document !== "undefined") {
  if (document.body) diagInit();
  else document.addEventListener("DOMContentLoaded", diagInit, { once: true });
}

/* ★ 版本水印 —— 在 DevTools Console 第一行就能看到。
 *   看到 "v2026-09-14-B plan-merge-active" = 新版已生效。
 *   看不到 = 浏览器用了缓存，Ctrl+Shift+R 强刷一下。
 *   （这行加在 IIFE 外，模块顶层，重复 import 也只会跑一次。） */
if (typeof console !== "undefined") {
  console.log("%c[h3-director] v2026-09-14-B plan-merge-active",
    "color:#4fff8f;font-weight:bold;");
}
