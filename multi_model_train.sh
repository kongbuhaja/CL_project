nohup python3 train.py --model DarkNet19 --gpus 0 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 150  > DarkNet19.log 2>&1 &
nohup python3 train.py --model ResNet18 --gpus 1 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 150 > ResNet18.log 2>&1 &
nohup python3 train.py --model VGG19 --gpus 2 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 150 > VGG19.log 2>&1 &
nohup python3 train.py --model GoogleNet22 --gpus 3 --channel 64 --epochs 200 --eval_term 5 --dataset mnist --batch_size 150 > GoogleNet22.log 2>&1 &

