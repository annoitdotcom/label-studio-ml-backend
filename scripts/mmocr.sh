label-studio-ml start mmocr_backend --with \
    layout_model_path=./weights/dclayout_model.pt \
    ocr_model_path=./weights/ocr_model.pt \
    device=cuda \
    -p 9091
