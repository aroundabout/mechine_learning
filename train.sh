export PYTHONPATH=$PYTHONPATH:~/mechine_learning
echo $PYTHONPATH

python runner/train.py --mode=train --dataset=CIFAR10 --model_name=deepercnn --loss=cross_entropy --device=0