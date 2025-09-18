for mode in train eval record; do
    for seed in 1 3 5; do
        for algo in ppo_explo td3_explo; do
            for task in PickCube StackCube PegInsert TakeLid OpenDrawer PutBox; do
                python scripts/run_llm_tale.py --env=$task --mode=$mode --seed=$seed --algo=$algo &
            done
            wait
        done
    done
done