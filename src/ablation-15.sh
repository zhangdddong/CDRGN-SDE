
# ICEWS05-15 w/o NSDE
#python main.py \
#-d ICEWS05-15 \
#--train-history-len 3 --test-history-len 3 --dilate-len 1 \
#--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
#--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
#--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
#--gpu 0 \
#--use-cd --use-time-decoder \
#--name 10_CDRGN-SDE


# ICEWS05-15 w/o CD
#python main.py \
#-d ICEWS05-15 \
#--train-history-len 3 --test-history-len 3 --dilate-len 1 \
#--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
#--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
#--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
#--gpu 0 \
#--use-sde --res --use-time-decoder \
#--name 11_CDRGN-SDE

# ICEWS05-15 w/o TD
python main.py \
-d ICEWS05-15 \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
--gpu 0 \
--use-cd --use-sde --res \
--name 12_CDRGN-SDE