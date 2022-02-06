CKPT=$1
FEAT_SUFFIX=$2
NL=$3

ARCH=iresnet${NL}
FEAT_PATH=./features/magface_${ARCH}/
mkdir -p ${FEAT_PATH}

CUDA_VISIBLE_DEVICES=1 python gen_feat_my.py --arch ${ARCH} \
                    --inf_list /mlcv/Databases/FACE_REG/EVAL/OUT_merge_split_resize_112/list_images_file.txt \
                    --feat_list ${FEAT_PATH}/OUT_merge_${FEAT_SUFFIX} \
                    --batch_size 1 \
                    --resume ${CKPT}


# CUDA_VISIBLE_DEVICES=1 python gen_feat.py --arch ${ARCH} \
#                     --inf_list toy_imgs/img.list \
#                     --feat_list ${FEAT_PATH}/OUT_merge_${FEAT_SUFFIX}.list \
#                     --batch_size 1 \
#                     --resume ${CKPT}