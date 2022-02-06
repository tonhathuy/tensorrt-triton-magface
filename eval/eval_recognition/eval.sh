CKPT=$1
FEAT_SUFFIX=$2
NL=$3

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

CUDA_VISIBLE_DEVICES=1 python ../../inference/gen_feat.py --arch ${ARCH} \
                    --inf_list data/lfw/img.list \
                    --feat_list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
                    --batch_size 512 \
                    --resume ${CKPT}

# CUDA_VISIBLE_DEVICES=1 python ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/cfp/img.list \
#                     --feat_list ${FEAT_PATH}/cfp_${FEAT_SUFFIX}.list \
#                     --batch_size 512 \
#                     --resume ${CKPT}

# CUDA_VISIBLE_DEVICES=1 python ../../inference/gen_feat.py --arch ${ARCH} \
#                     --inf_list data/agedb/img.list \
#                     --feat_list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
#                     --batch_size 512 \
#                     --resume ${CKPT}

# echo evaluate lfw
# CUDA_VISIBLE_DEVICES=1 python eval_1v1.py \
#         --feat-list ${FEAT_PATH}/lfw_${FEAT_SUFFIX}.list \
# 		--pair-list data/lfw/pair.list \


# echo evaluate cfp
# CUDA_VISIBLE_DEVICES=1 python eval_1v1.py \
#         --feat-list ${FEAT_PATH}/cfp_${FEAT_SUFFIX}.list \
# 		--pair-list data/cfp/pair.list \

# echo evaluate agedb
# CUDA_VISIBLE_DEVICES=1 python eval_1v1.py \
#         --feat-list ${FEAT_PATH}/agedb_${FEAT_SUFFIX}.list \
# 		--pair-list data/agedb/pair.list \