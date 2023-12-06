python3 train.py --model DarkNet19 --channel 16 --epochs 100
python3 train.py --model ResNet18 --channel 16 --epochs 100
python3 train.py --model VGG19 --channel 16 --epochs 100
python3 train.py --model GoogleNet22 --channel 16 --epochs 100

python3 train.py --model DarkNet19 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
python3 train.py --model ResNet18 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
python3 train.py --model VGG19 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
python3 train.py --model GoogleNet22 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
 
# python3 train.py --model DarkNet19 --channel 64 --epochs 100 --eval_term 10 --dataset cifar100 --batch_size 100 
# python3 train.py --model ResNet18 --channel 64 --epochs 100 --eval_term 10 --dataset cifar100 --batch_size 100
# python3 train.py --model VGG19 --channel 64 --epochs 100 --eval_term 10 --dataset cifar100 --batch_size 100
# python3 train.py --model GoogleNet22 --channel 64 --epochs 100 --eval_term 10 --dataset cifar100 --batch_size 100

# python3 train.py --model DarkNet19 --gpus 0 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model ResNet18 --gpus 1 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model VGG19 --gpus 2 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model GoogleNet22 --gpus 3 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
