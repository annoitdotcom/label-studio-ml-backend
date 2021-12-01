label-studio-ml start mmdet_backend --with \
    config_file=./submodule_mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py \
    checkpoint_file=./weights/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    device=cuda \
    -p 9091
