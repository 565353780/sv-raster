# 小妖怪头
SHAPE_ID="003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
# 女人上半身
SHAPE_ID="017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
# 长发男人头
SHAPE_ID="0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

DATA_FOLDER="${HOME}/chLi/Dataset/pixel_align/${SHAPE_ID}"

CUDA_VISIBLE_DEVICES=2 \
  python train.py \
  --eval \
  --source_path ${DATA_FOLDER}/colmap/ \
  --model_path ${DATA_FOLDER}/svraster/ \
  --white_background \
  --bound_mode camera_median \
  --lambda_T_inside 0.01 \
  --lambda_T_concen 0.1 \
  --lambda_normal_dmean 0.001 \
  --lambda_ascending 0.01

#python extract_mesh.py \
#${DATA_FOLDER}/svraster/
