# running PPO
python3 main.py --use-gae --num-mini-batch 4 --use-linear-lr-decay --num-env-steps 10000000 --env-name Abi --log-dir ./log/test --demand-matrix Abi_500.txt --model-save-path ./model/test 


