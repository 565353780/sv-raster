DATA_FOLDER=$HOME/chLi/Dataset/GS/haizei_1

python train.py \
  --eval \
  --source_path ${DATA_FOLDER}/gs/ \
  --model_path ${DATA_FOLDER}/svraster/ \
  --bound_mode camera_median \
  --lambda_T_inside 0.01 \
  --lambda_normal_dmean 0.001 \
  --lambda_ascending 0.01 \
  --lambda_sparse_depth 0.01

python extract_mesh.py \
  ${DATA_FOLDER}/svraster/
