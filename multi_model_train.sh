nohup python3 train.py --model DarkNet19 --gpus 0 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 150 --load True > DarkNet19.log 2>&1 &
nohup python3 train.py --model ResNet18 --gpus 1 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 150 --load True > ResNet18.log 2>&1 &
nohup python3 train.py --model VGG19 --gpus 2 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 150 --load True > VGG19.log 2>&1 &
nohup python3 train.py --model GoogleNet22 --gpus 3 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 150 --load True > GoogleNet22.log 2>&1 &

