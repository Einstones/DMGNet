# Dynamic Mixed-order Laplacian Deep Network for Image Restoration
<hr>
<i>In this paper, we propose an iterative framework with Dynamic Mixed-order Graph Laplacian learning (DMGNet) for image restoration. Specifically, without loss of interpretability, we integrate a Routing Laplacian Relation Module into the degradation mapping functions to flexibly estimate the complicated degradation process. We design the Adaptive Structure-Guided Module to harness the structure information to modulate the reconstruction process for better details preservation. By integrating the flexible mixed-order laplacian function and structured-preserving modulation, we unfold the iterative process into a trainable DNN. We have validated the effectiveness of the proposed method on six image restoration tasks. The experimental results demonstrate its remarkable performance in terms of both quantitative metrics as well as visual quality. Particularly, it supports the satisfied interpretability, and has potential to preserve the subtle image structures and textures..</i>

## Package dependencies
The project is built with PyTorch 1.7.1, Python3.7, CUDA10.1. For package dependencies, you can install them by:
```bash
pip3 install -r requirements.txt
```

## Data preparation 
### Denoising
For training data of SIDD, you can download the SIDD-Medium dataset from the [official url](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php).
Then generate training patches for training by:
```python
python3 generate_patches_SIDD.py --src_dir ../SIDD_Medium_Srgb/Data
```
For evaluation, we use the same evaluation data as [here](https://drive.google.com/drive/folders/1j5ESMU0HJGD-wU6qbEdnt569z7sM3479), and put it into the dir `../datasets/denoising/sidd/val`.

## Training
### Denoising
To train `DMGNet` on SIDD, we use 4 V100 GPUs (80G) and run for 500 epochs:

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,  python -m torch.distributed.launch --master_port 50000 --nproc_per_node=2  ../Denoise/train.py
```

More configuration can be founded in `train.py`.


### Deraining

To train `DMGNet` on Deraining Task, you can run:

```python
CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --master_port 50004 --nproc_per_node=2  ../Derain/train.py
```

### DeBluring

To train `DMGNet` on Deraining Task, you can run:

```python
CUDA_VISIBLE_DEVICES=2,3  python -m torch.distributed.launch --master_port 50005 --nproc_per_node=2  ../Deblur/train.py
```

## Acknowledgement

This code borrows heavily from [MIRNet](https://github.com/swz30/MIRNet) and [Uformer](https://github.com/ZhendongWang6/Uformer).


## Contact
Please contact us if there is any question or suggestion(hanyudong.sdu@gmail.com).
