nohup python3 train.py --model DarkNet19 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 100 --gpus 1 > train.log 2>&1 &
# python3 train.py --model ResNet18 --channel 64 --epochs 200 --eval_term 10 --dataset imagenet --batch_size 100
# python3 train.py --model VGG19 --channel 64 --epochs 200 --eval_term 10 --dataset imagenet --batch_size 100
# python3 train.py --model GoogleNet22 --channel 64 --epochs 200 --eval_term 10 --dataset imagenet --batch_size 100
