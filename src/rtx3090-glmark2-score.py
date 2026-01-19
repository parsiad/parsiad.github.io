# %% [raw]
# +++
# aliases = [
#   "/blog/2021/rtx3090_glmark2_score"
# ]
# date = 2021-02-08
# title = "Gigabyte RTX 3090 glmark2 score"
# +++

# %% tags=["no_cell"]
from _boilerplate import init

init()

# %% [markdown]
# ```
# =======================================================
#     OpenGL Information
#     GL_VENDOR:     NVIDIA Corporation
#     GL_RENDERER:   GeForce RTX 3090/PCIe/SSE2
#     GL_VERSION:    4.6.0 NVIDIA 460.39
# =======================================================
# [build] use-vbo=false:
#  FPS: 4254 FrameTime: 0.235 ms
# [build] use-vbo=true: FPS: 34143 FrameTime: 0.029 ms
# [texture] texture-filter=nearest: FPS: 36263 FrameTime: 0.028 ms
# [texture] texture-filter=linear: FPS: 36188 FrameTime: 0.028 ms
# [texture] texture-filter=mipmap: FPS: 36095 FrameTime: 0.028 ms
# [shading] shading=gouraud: FPS: 34701 FrameTime: 0.029 ms
# [shading] shading=blinn-phong-inf: FPS: 34514 FrameTime: 0.029 ms
# [shading] shading=phong: FPS: 34378 FrameTime: 0.029 ms
# [shading] shading=cel: FPS: 34343 FrameTime: 0.029 ms
# [bump] bump-render=high-poly: FPS: 26226 FrameTime: 0.038 ms
# [bump] bump-render=normals: FPS: 36662 FrameTime: 0.027 ms
# [bump] bump-render=height: FPS: 36032 FrameTime: 0.028 ms
# [effect2d] kernel=0,1,0;1,-4,1;0,1,0;: FPS: 34233 FrameTime: 0.029 ms
# [effect2d] kernel=1,1,1,1,1;1,1,1,1,1;1,1,1,1,1;: FPS: 30708 FrameTime: 0.033 ms
# [pulsar] light=false:quads=5:texture=false: FPS: 34857 FrameTime: 0.029 ms
# [desktop] blur-radius=5:effect=blur:passes=1:separable=true:windows=4: FPS: 5131 FrameTime: 0.195 ms
# [desktop] effect=shadow:windows=4: FPS: 6281 FrameTime: 0.159 ms
# [buffer] columns=200:interleave=false:update-dispersion=0.9:update-fraction=0.5:update-method=map: FPS: 856 FrameTime: 1.168 ms
# [buffer] columns=200:interleave=false:update-dispersion=0.9:update-fraction=0.5:update-method=subdata: FPS: 1118 FrameTime: 0.894 ms
# [buffer] columns=200:interleave=true:update-dispersion=0.9:update-fraction=0.5:update-method=map: FPS: 1147 FrameTime: 0.872 ms
# [ideas] speed=duration: FPS: 9870 FrameTime: 0.101 ms
# [jellyfish] <default>: FPS: 28603 FrameTime: 0.035 ms
# [terrain] <default>: FPS: 2025 FrameTime: 0.494 ms
# [shadow] <default>: FPS: 17967 FrameTime: 0.056 ms
# [refract] <default>: FPS: 12466 FrameTime: 0.080 ms
# [conditionals] fragment-steps=0:vertex-steps=0: FPS: 34216 FrameTime: 0.029 ms
# [conditionals] fragment-steps=5:vertex-steps=0: FPS: 34106 FrameTime: 0.029 ms
# [conditionals] fragment-steps=0:vertex-steps=5: FPS: 34152 FrameTime: 0.029 ms
# [function] fragment-complexity=low:fragment-steps=5: FPS: 34183 FrameTime: 0.029 ms
# [function] fragment-complexity=medium:fragment-steps=5: FPS: 34032 FrameTime: 0.029 ms
# [loop] fragment-loop=false:fragment-steps=5:vertex-steps=5: FPS: 34090 FrameTime: 0.029 ms
# [loop] fragment-steps=5:fragment-uniform=false:vertex-steps=5: FPS: 34116 FrameTime: 0.029 ms
# [loop] fragment-steps=5:fragment-uniform=true:vertex-steps=5: FPS: 33864 FrameTime: 0.030 ms
# =======================================================
#                                   glmark2 Score: 25509
# =======================================================
# ```
