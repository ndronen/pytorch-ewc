#!/bin/bash  

# --hidden-size HIDDEN_SIZE=400
# --hidden-layer-num HIDDEN_LAYER_NUM=2
# --hidden-dropout-prob HIDDEN_DROPOUT_PROB=0.5
# --input-dropout-prob INPUT_DROPOUT_PROB=0.2
# --task-number TASK_NUMBER
# --epochs-per-task EPOCHS_PER_TASK
# --lamda LAMDA
# --lr LR
# --weight-decay WEIGHT_DECAY
# --batch-size BATCH_SIZE
# --fisher-estimation-sample-size FISHER_ESTIMATION_SAMPLE_SIZE
# --eval-log-interval EVAL_LOG_INTERVAL
# --consolidate

for epochs_per_task in 1 3 10
do
    for lamda in 1 3 10 30 100
    do
        for opt_name in "sgd" "adam"
        do
            python main.py --epochs-per-task $epochs_per_task --lamda $lamda --opt_name $opt_name --consolidate
        done
    done
done
