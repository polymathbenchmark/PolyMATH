# # python annotation_utils.py \
# # --mode create_metadata \
# # --paper_path "../ntse_15_16_papers/QP_SAT_LANG_MAT_Bihar_NTSE_Stage-1_2015-16.pdf" \
# # --annotator_name ujjwalaa \
# # --datastore_path "../datastore"


# python annotation_utils.py \
# --mode create_annotation \
# --datastore_path "../datastore"
# # Run this script from the root directory as
# # bash scripts/annotation_utils.sh


## Metadata creation & initial setup
#ANNOTATOR_NAME=
#python ./scripts/annotation_utils.py \
#--mode create_metadata \
#--paper_path "./raw_dataset/13_14_papers/NTSE-Stage-1-2013-Paper-MAT(Rajasthan).pdf" \
#--annotator_name "$ANNOTATOR_NAME" \
#--datastore_path "./datastore"

## Single paper merge screenshots tool
#python ./scripts/annotation_utils.py \
#--mode merge_screenshots \
#--paper_path "./datastore/QP_ANDHRA PRADESH_NTSE_STAGE 1_SAT_2019-20"

## Bulk merge screenshots tool
#python ./scripts/annotation_utils.py \
#--mode merge_screenshots \
#--datastore_path "./datastore"

## Single paper annotation creation
python ./scripts/annotation_utils.py \
--mode create_annotation \
--paper_path "./datastore/QP_BIHAR_NTSE_STAGE 1_SAT_2019-20" \
#--overwrite

## Bulk annotation creation tool
# python ./scripts/annotation_utils.py \
# --mode create_annotation \
# --datastore_path "./datastore" \
# --overwrite

### Single paper answer key merging
#python ./scripts/annotation_utils.py \
#--mode merge_answer_keys \
#--paper_path "./datastore/QP_West Bengal NTSE Stage 1 2017-18 SAT"

### Bulk answer key merging
# python ./scripts/annotation_utils.py \
# --mode merge_answer_keys \
# --datastore_path "./datastore" \

## Freeze annotation
#TO BE ADDED
