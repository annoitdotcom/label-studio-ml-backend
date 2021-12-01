label-studio-ml start mmocr_backend --with \
    det_config=./submodule_mmocr/configs/textdet/dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py \
    det_ckpt=./weights/dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth \
    recog_config=./submodule_mmocr/configs/textrecog/seg/seg_r31_1by16_fpnocr_academic.py \
    recog_ckpt=./weights/seg_r31_1by16_fpnocr_academic-72235b11.pth
    device=cuda \
    -p 9091
