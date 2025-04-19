# python script/waymo/waymo_converter.py \
#         --root_dir '/media/ml4u/ExtremeSSD/datasets/waymo/raw' \
#         --save_dir '/media/ml4u/ExtremeSSD/datasets/waymo/processed' \
#         --split_file script/waymo/waymo_splits/train_static_short.txt \
#         --segment_file script/waymo/waymo_splits/segment_list_train.txt

python script/waymo/waymo_converter.py \
        --root_dir '/media/ml4u/ExtremeSSD/datasets/waymo/raw' \
        --save_dir '/media/ml4u/ExtremeSSD/datasets/waymo/processed' \
        --split_file script/waymo/waymo_splits/val_dynamic.txt \
        --segment_file script/waymo/waymo_splits/segment_list_val.txt \
        --track_file result.json