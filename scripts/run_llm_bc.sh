for mode in collect train eval; do
    for task in PickCube StackCube PegInsert; do
        for seed in 0 2 4; do
            python scripts/prepare_llm_bc.py --env=$task --mode=$mode --seed=$seed &
        done
    done
    wait
done

for mode in train eval record; do
    for algo in ppo_explo td3_explo; do
        for task in PickCube StackCube PegInsert; do
            python scripts/run_llm_tale.py --env=$task --mode=$mode --seed 1 3 5 --algo=$algo --llm_bc &
        done
        wait
    done
done
