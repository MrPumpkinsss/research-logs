# **2026/3/19 TODO**


## 1. （已解决）通过build from source重新部署虚拟机环境
原先是基于docker compose部署的一个ray header，两个ray worker的容器，源代码通过配置容器时候的yml文件中的volume传入，不需要build就可以传入一两个修改后的py文件，随着修改的源文件越来越多，这个方法会变得很麻烦。于是通过uv pip install .e重新管理容器。

## 2. （已解决）解决vllm分割不同数量layers到不同container后，vllm V1引擎初始化不稳定问题
部署Qwen3-0.6B有概率报错，启动指令
```
python3 -m vllm.entrypoints.openai.api_server     --model Qwen/Qwen3-0.6B     --pipeline-parallel-size 3     --distributed-executor-backend ray     --max-model-len 1024     --gpu-memory-utilization 0.3
```
```
(APIServer pid=1370) raise RuntimeError(
(APIServer pid=1370) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```
```
(EngineCore_DP0 pid=1394) ERROR 03-19 11:07:19 [core.py:1006] File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 5346, in get_attn_backends_for_group
(EngineCore_DP0 pid=1394) ERROR 03-19 11:07:19 [core.py:1006] layers = get_layers_from_vllm_config(
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
虽然可以通过--enforce-eager不编译图来避免但是不妥
需要改 vllm/v1/worker/gpu_model_runner.py的get_attn_backends_for_group函数。
kv_cache_group_spec.layer_names 里是全局所有层，需要跳过不属于自己的层名

改后报错
```
(EngineCore pid=311) (RayWorkerWrapper pid=144, ip=172.18.0.2) ERROR 03-19 19:10:25 [ray_utils.py:74] kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
(EngineCore pid=311) (RayWorkerWrapper pid=144, ip=172.18.0.2) ERROR 03-19 19:10:25 [ray_utils.py:74] ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=311) (RayWorkerWrapper pid=144, ip=172.18.0.2) ERROR 03-19 19:10:25 [ray_utils.py:74] IndexError: list index out of range
```
需要改\vllm\model_executor\layers\attention\attention.py的get_attention_context函数
编译的图里硬编码了全量 layer_name，但当前 PP rank只加载了自己分配的层

vllm官方镜像mistral-common过旧，需要uv pip install --system --upgrade mistral-common

现在已经能稳定的给不同的容器分配不同的层数了，但是目前只验证了容器是单机单卡的情况

## 3. 测试并理解AMCoEdge

https://github.com/ChangfuXu/AMCoEdge

代码已经能跑通，但是源代码用的是tensorflow 1.4.0非常老。看看能不能把AMCoEdge原理用到vllm的自适应分配上。

# **论文阅读**

Attention Residuals很火
还在读