example for model=slow_r50 and batch_size=4, from torchinfo
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
Net                                                [4, 3]                    --
├─ModuleList: 1-1                                  --                        --
│    └─ResNetBasicStem: 2-1                        [4, 64, 8, 56, 56]        --
│    │    └─Conv3d: 3-1                            [4, 64, 8, 112, 112]      9,408
│    │    └─BatchNorm3d: 3-2                       [4, 64, 8, 112, 112]      128
│    │    └─ReLU: 3-3                              [4, 64, 8, 112, 112]      --
│    │    └─MaxPool3d: 3-4                         [4, 64, 8, 56, 56]        --
│    └─ResStage: 2-2                               [4, 256, 8, 56, 56]       --
│    │    └─ModuleList: 3-5                        --                        215,808
│    └─ResStage: 2-3                               [4, 512, 8, 28, 28]       --
│    │    └─ModuleList: 3-6                        --                        1,219,584
│    └─ResStage: 2-4                               [4, 1024, 8, 14, 14]      --
│    │    └─ModuleList: 3-7                        --                        9,981,952
│    └─ResStage: 2-5                               [4, 2048, 8, 7, 7]        --
│    │    └─ModuleList: 3-8                        --                        20,207,616
│    └─ResNetBasicHead: 2-6                        [4, 3]                    --
│    │    └─AvgPool3d: 3-9                         [4, 2048, 1, 1, 1]        --
│    │    └─Dropout: 3-10                          [4, 2048, 1, 1, 1]        --
│    │    └─Linear: 3-11                           [4, 1, 1, 1, 3]           6,147
│    │    └─AdaptiveAvgPool3d: 3-12                [4, 3, 1, 1, 1]           --
====================================================================================================
Total params: 31,640,643
Trainable params: 31,640,643
Non-trainable params: 0
Total mult-adds (G): 166.96
====================================================================================================
Input size (MB): 19.27
Forward/backward pass size (MB): 5690.36
Params size (MB): 126.56
Estimated Total Size (MB): 5836.19
====================================================================================================



count of weights: 31,640,643
weights size: 31,640,643 * 4 (bytes) = 126,562,572 (bytes) = 120.7 (MB)
gradients size: 120.7 (MB)
optimizer state (AdamW: m + v): 31,640,643 * 4 (bytes) * 2 = 253,125,144 (bytes) = 241.4 (MB)
total independent of batch_size parameters size: 482.8 (MB) 
size of activations for batch_size=4: 5,690 (MB), from torchinfo
size of activations for batch_size=8: 11,381 (MB), from torchinfo
size of activations for batch_size=16: 22,761 (MB), from torchinfo
FLOPs by one example: 41.74 (GFLOPs) 

For RTX5090 with 32GB VRAM we can use batch_size=16, because total GPU memory usage during training is 23,243.8 MB = 22.7GB, it takes 70.9% of all memory space