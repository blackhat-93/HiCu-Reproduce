@echo off

:: Edit value of num_runs to change the number of repeat runs
SET num_runs=3

FOR /L %%i IN (1,1,%num_runs%) DO (
    ECHO Starting run No. %%i of %num_runs%

    python main.py ^
        --model MultiResCNN ^
        --vocab .\data\mimic3\vocab.csv ^
        --decoder HierarchicalHyperbolic ^
        --loss ASL ^
        --asl_config "1,0,0.03" ^
        --Y 50 ^
        --data_path .\data\mimic3\train_50.csv ^
        --MAX_LENGTH 4096 ^
        --embed_file .\data\mimic3\processed_full_100.embed ^
        --tune_wordemb ^
        --batch_size 8 ^
        --lr 5e-5 ^
        --n_epochs "2,2,3,5,50" ^
        --criterion prec_at_8 ^
        --random_seed 0 ^
        --num_workers 1
)

ECHO Completed all %num_runs% runs