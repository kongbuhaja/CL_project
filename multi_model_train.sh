nohup python3 train.py --model DarkNet19 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 100 --gpus 0 > DarkNet19.log 2>&1 &
nohup python3 train.py --model ResNet18 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 100 --gpus 1 > ResNet18.log 2>&1 &
nohup python3 train.py --model VGG19 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 150 --gpus 2 > VGG19.log 2>&1 &
nohup python3 train.py --model GoogleNet22 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 100 --gpus 3 > GoogleNet22.log 2>&1 &

