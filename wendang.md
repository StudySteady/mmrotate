python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_trainval.json

python tools/data/dota/split/img_split.py --base-json \
  tools/data/dota/split/split_configs/ss_test.json
裁剪 来自 https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md



测试rotated_retinanet
+--------------------+--------+--------+--------+-------+
| class              | gts    | dets   | recall | ap    |
+--------------------+--------+--------+--------+-------+
| plane              | 18855  | 57135  | 0.960  | 0.907 |
| baseball-diamond   | 1080   | 18819  | 0.948  | 0.873 |
| bridge             | 4240   | 86671  | 0.740  | 0.544 |
| ground-track-field | 750    | 28442  | 0.935  | 0.805 |
| small-vehicle      | 295272 | 443172 | 0.385  | 0.299 |
| large-vehicle      | 54168  | 321074 | 0.793  | 0.633 |
| ship               | 88881  | 216829 | 0.832  | 0.782 |
| tennis-court       | 6046   | 37943  | 0.977  | 0.908 |
| basketball-court   | 1226   | 14446  | 0.921  | 0.839 |
| storage-tank       | 14192  | 90509  | 0.797  | 0.703 |
| soccer-ball-field  | 829    | 19140  | 0.812  | 0.657 |
| roundabout         | 1050   | 31946  | 0.700  | 0.430 |
| harbor             | 15542  | 115171 | 0.784  | 0.644 |
| swimming-pool      | 4882   | 44758  | 0.861  | 0.767 |
| helicopter         | 1214   | 39633  | 0.850  | 0.784 |
+--------------------+--------+--------+--------+-------+
| mAP                |        |        |        | 0.705 |
+--------------------+--------+--------+--------+-------+
{'mAP': 0.7049340605735779}


eval_from_pkl.py 通过pkl跑mAP
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 18788 | 54689  | 0.961  | 0.907 |
| baseball-diamond   | 1087  | 18172  | 0.941  | 0.873 |
| bridge             | 4181  | 83807  | 0.747  | 0.554 |
| ground-track-field | 733   | 27475  | 0.941  | 0.821 |
| small-vehicle      | 58868 | 385134 | 0.886  | 0.734 |
| large-vehicle      | 43075 | 294532 | 0.897  | 0.726 |
| ship               | 76153 | 210256 | 0.892  | 0.786 |
| tennis-court       | 5923  | 36765  | 0.983  | 0.908 |
| basketball-court   | 1180  | 13801  | 0.940  | 0.859 |
| storage-tank       | 13670 | 86565  | 0.811  | 0.745 |
| soccer-ball-field  | 827   | 18212  | 0.822  | 0.680 |
| roundabout         | 973   | 31125  | 0.933  | 0.822 |
| harbor             | 15468 | 113347 | 0.782  | 0.644 |
| swimming-pool      | 3836  | 43800  | 0.933  | 0.836 |
| helicopter         | 1189  | 39264  | 0.877  | 0.803 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.780 |
+--------------------+-------+--------+--------+-------+
Evaluation result:
{'mAP': 0.7798087000846863}



(mmrotate) liuxinjia@LAPTOP-I1NLFAFF:~/MMRotate/mmrotate$ python tools/train.py \
configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py \
--work-dir work_dirs/rotated_retinanet_le90_smoke \
--cfg-options \
data.samples_per_gpu=1 \
data.workers_per_gpu=0 \
runner.max_epochs=1 \
checkpoint_config.interval=1 \
evaluation.interval=1
/home/liuxinjia/miniconda3/envs/mmrotate/lib/python3.8/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
2026-03-30 17:56:20,994 - mmrotate - INFO - Environment info:
------------------------------------------------------------
sys.platform: linux
Python: 3.8.20 (default, Oct  3 2024, 15:24:27) [GCC 11.2.0]
CUDA available: True
GPU 0: NVIDIA GeForce RTX 3070 Laptop GPU
CUDA_HOME: /usr/local/cuda-11.7
NVCC: Cuda compilation tools, release 11.7, V11.7.64
GCC: x86_64-conda_cos7-linux-gnu-gcc (Anaconda gcc) 11.2.0
PyTorch: 1.13.1+cu117
PyTorch compiling details: PyTorch built with:
  - GCC 9.3
  - C++ Version: 201402
  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.6.0 (Git Hash 52b5f107dd9cf10910aaa19cb47f3abf9b349815)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: AVX2
  - CUDA Runtime 11.7
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
  - CuDNN 8.5
  - Magma 2.6.1
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS= -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.13.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

TorchVision: 0.14.1+cu117
OpenCV: 4.12.0
MMCV: 1.7.0
MMCV Compiler: GCC 9.3
MMCV CUDA Compiler: 11.7
MMRotate: 0.3.4+b030f38
------------------------------------------------------------


2026-03-30 19:13:37,229 - mmrotate - INFO - Epoch [1][12750/12799]      lr: 2.500e-03, eta: 0:00:17, time: 0.362, data_time: 0.045, memory: 2119, loss_cls: 0.6937, loss_bbox: 0.8548, loss: 1.5485, grad_norm: 6.8606
2026-03-30 19:13:53,419 - mmrotate - INFO - Saving checkpoint at 1 epochs
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 12800/12800, 4.1 task/s, elapsed: 3135s, ETA:     0s/home/liuxinjia/miniconda3/envs/mmrotate/lib/python3.8/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/liuxinjia/miniconda3/envs/mmrotate/lib/python3.8/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/liuxinjia/miniconda3/envs/mmrotate/lib/python3.8/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
/home/liuxinjia/miniconda3/envs/mmrotate/lib/python3.8/site-packages/mmcv/__init__.py:20: UserWarning: On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process and add a data transformation module. In addition, it will rename the package names mmcv to mmcv-lite and mmcv-full to mmcv. See https://github.com/open-mmlab/mmcv/blob/master/docs/en/compatibility.md for more details.
  warnings.warn(
2026-03-30 20:06:29,457 - mmrotate - INFO - 
+--------------------+-------+--------+--------+-------+
| class              | gts   | dets   | recall | ap    |
+--------------------+-------+--------+--------+-------+
| plane              | 18788 | 126662 | 0.843  | 0.778 |
| baseball-diamond   | 1087  | 79125  | 0.743  | 0.024 |
| bridge             | 4181  | 172645 | 0.254  | 0.122 |
| ground-track-field | 733   | 33001  | 0.357  | 0.003 |
| small-vehicle      | 58868 | 332736 | 0.538  | 0.206 |
| large-vehicle      | 43075 | 421184 | 0.482  | 0.112 |
| ship               | 76153 | 785499 | 0.459  | 0.155 |
| tennis-court       | 5923  | 55981  | 0.810  | 0.727 |
| basketball-court   | 1180  | 16194  | 0.385  | 0.016 |
| storage-tank       | 13670 | 256151 | 0.553  | 0.299 |
| soccer-ball-field  | 827   | 18910  | 0.193  | 0.002 |
| roundabout         | 973   | 133390 | 0.762  | 0.026 |
| harbor             | 15468 | 316652 | 0.285  | 0.012 |
| swimming-pool      | 3836  | 156269 | 0.552  | 0.014 |
| helicopter         | 1189  | 26711  | 0.041  | 0.000 |
+--------------------+-------+--------+--------+-------+
| mAP                |       |        |        | 0.166 |
+--------------------+-------+--------+--------+-------+
2026-03-30 20:06:30,692 - mmrotate - INFO - Exp name: rotated_retinanet_obb_r50_fpn_1x_dota_le90.py
2026-03-30 20:06:30,692 - mmrotate - INFO - Epoch(val) [1][12800]       mAP: 0.1664