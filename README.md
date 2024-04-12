
# VIT-fine-tuning-with-ColossalAI-by-CM

This project reproduces the ViT (Vision Transformer) example from ColossalAI. It explores different training strategies with two image sizes (224 and 384). The strategies include "torch_ddp", "torch_ddp_fp16", "low_level_zero", and "gemini".

## Installation

Clone the repository and navigate to the ViT example directory:

```bash
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI/examples/images/vit
```

Run the demonstration script:

```bash
bash run_demo.sh
```
During this process, we will finetuned the VIT base on Beans Dataset(https://huggingface.co/datasets/AI-Lab-Makerere/beans)

## Configuration

The basic configuration settings used are:

- TP_SIZE: 2 (Tensor parallel size)
- PP_SIZE: 2 (Pipeline parallel size)
- GPUNUM: 1 (Number of GPUs)
- BS: 32 (Batch size per data parallel group)
- LR: 1e-4 (Learning rate)
- EPOCH: 3 (Number of epochs)
- WEIGHT_DECAY: 0.05
- WARMUP_RATIO: 0.3

These settings are fixed during the fine-tuning process.

### Hyperparameters

For the model and plugin configurations, the options are:

- MODEL: "google/vit-base-patch16-224" or "google/vit-base-patch16-384"
- PLUGIN: "torch_ddp", "torch_ddp_fp16", "low_level_zero", "gemini"

## Experimental Output

Here are the outputs for each configuration after every epoch:

| Size | Strategy        | Epoch 1                              | Epoch 2                              | Epoch 3                              |
|------|-----------------|--------------------------------------|--------------------------------------|--------------------------------------|
| 224  | torch_ddp       | average_loss=0.1712, accuracy=0.9531 | average_loss=0.0359, accuracy=0.9844 | average_loss=0.0319, accuracy=0.9922 |
| 224  | torch_ddp_fp16  | average_loss=0.1436, accuracy=0.9531 | average_loss=0.0311, accuracy=0.9922 | average_loss=0.0250, accuracy=0.9922 |
| 224  | low_level_zero  | average_loss=0.1527, accuracy=0.9453 | average_loss=0.0284, accuracy=0.9922 | average_loss=0.0183, accuracy=0.9922 |
| 224  | gemini          | average_loss=0.1685, accuracy=0.9375 | average_loss=0.0339, accuracy=0.9844 | average_loss=0.0254, accuracy=0.9844 |
| 384  | torch_ddp       | average_loss=0.0578, accuracy=0.9766 | average_loss=0.0477, accuracy=0.9844 | average_loss=0.0095, accuracy=1.0000 |
| 384  | torch_ddp_fp16  | average_loss=0.0598, accuracy=0.9766 | average_loss=0.0347, accuracy=0.9844 | average_loss=0.0074, accuracy=1.0000 |
| 384  | low_level_zero  | average_loss=0.0578, accuracy=0.9766 | average_loss=0.0306, accuracy=0.9844 | average_loss=0.0105, accuracy=0.9922 |
| 384  | gemini          | average_loss=0.0538, accuracy=0.9688 | average_loss=0.0417, accuracy=0.9844 | average_loss=0.0095, accuracy=1.0000 |
