
# ICEWS14s
python main.py \
-d ICEWS14s \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
--gpu 0 \
--use-cd --use-sde --res --use-time-decoder \
--name 01_CDRGN-SDE

# ICEWS05-15
python main.py \
-d ICEWS05-15 \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
--gpu 0 \
--use-cd --use-sde --res --use-time-decoder \
--name 02_CDRGN-SDE

# ICEWS18
python main.py \
-d ICEWS18 \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
--gpu 0 \
--use-cd --use-sde --res --use-time-decoder \
--name 03_CDRGN-SDE

# GDELT
python main.py \
-d GDELT \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 \
--gpu 0 \
--use-cd --use-sde --res --use-time-decoder \
--name 04_CDRGN-SDE

# WIKI
python main.py \
-d WIKI \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 \
--use-sde --res --use-cd --use-time-decoder \
--name 05_CDRGN-SDE

# YAGO
python main.py \
-d YAGO \
--train-history-len 3 --test-history-len 3 --dilate-len 1 \
--lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
--decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
--entity-prediction --relation-prediction --angle 10 --discount 1 --task-weight 0.7 \
--use-sde --res --use-cd --use-time-decoder \
--name 06_CDRGN-SDE