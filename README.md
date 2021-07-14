# multi_gpu_image_feature_extractor
pytorch;multi gpu;image feature;

This is an implementation of extractig feature from image with resnet(pretrained on imagenet or place365) or swin_transformer.

## run
Download the resnet paras pretrained in imagenet [resnet_imagenet](https://pan.baidu.com/s/1aK5hiUF05XQewToxU7BLqg) (psw:ptal)or resnet paras pretrained in place365 [resnet_place365](https://pan.baidu.com/s/1GKPLvZlPe5O0aXJ6uEJLQQ) (psw:i0d8)
run the code by:
```
cd multi_gpu_image_feature_extractor
sh extract.sh
```
or:
```
python extrat_image_feature_from_image.py \
    --imagelist_file your_image_file_list \
    --out_path_root your_featurefile_save_path \
    --modelname resnet_imagenet \
    -b 48
```
you can set the modelname as "resnet_imagenet", "resnet_place365" or "swin_base_patch4_window12_384" so far
