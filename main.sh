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

cuda_device=-1

for epochs_per_task in 1 2 3 6 10
do
    for lamda in 1 3 5 10 30 50
    do
        for opt_name in "sgd" "adam"
        do
            for lr in 1e-3 1e-4 1e-5
            do
                if [[ cuda_device -eq 7 ]]
                then
                    cuda_device=0
                else
                    cuda_device=$((cuda_device + 1))
                fi
                echo CUDA_VISIBLE_DEVICES=$cuda_device python main.py --epochs-per-task $epochs_per_task --lamda $lamda --opt-name $opt_name --lr $lr --consolidate
            done
        done
    done
done
