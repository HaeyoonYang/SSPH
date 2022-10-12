# Self-supervised Pretraining for Deep Hash-based Image Retrieval
Official code for "Self-supervised Pretraining for Deep Hash-based Image Retrieval" (ICIP 2022)


## Requirements
```
    pip install requirements.txt
```

## Datasets
Download datasets: 
[ImageNet](https://drive.google.com/file/d/1X5iZNRM7wZ1eXz-ZCgNCcxpb913UiYNK/view?usp=sharing) 
[NUS-WIDE](https://drive.google.com/file/d/1TAjFKnOEse4xU_ScZOM8NgQLGexebmRn/view?usp=sharing) 
[MS COCO](https://drive.google.com/file/d/1EsRZP3YsLbkbJ9rNXA4x5BFkHVFIGlQP/view?usp=sharing)

## Code Execution
### Training
```
    python train.py --gpu_id=0 --dataset=nuswide --encoder=self-supervised --N_bits=64 --transformation_scale=0.5 --loss_type=DHD
```
(Change arguments before the execution)


### Testing
Trained model samples are [here](https://drive.google.com/drive/folders/1MkMa0cKVSQrQHTkfoKcyiiGyG8cjdpsj?usp=sharing)
```
    python train.py --gpu_id=0 --dataset=nuswide --encoder=self-supervised --N_bits=64 --model_path=models/selfsupervised_nuswide_64_0.5_0.1.pth
```
(Change arguments before the execution. Set --dataset, --encoder, --N_bits the same as the trained model)

Outputs top-10 retrieved images & mAP score.
Visualized results are saved at 'Vis/'

