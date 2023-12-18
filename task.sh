# python3 train.py --model DarkNet19 --channel 16 --epochs 100
# python3 train.py --model ResNet18 --channel 16 --epochs 100
# python3 train.py --model VGG19 --channel 16 --epochs 100
# python3 train.py --model GoogleNet22 --channel 16 --epochs 100

# python3 train.py --model DarkNet19 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
# python3 train.py --model ResNet18 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
# python3 train.py --model VGG19 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
# python3 train.py --model GoogleNet22 --channel 16 --epochs 200 --eval_term 5 --dataset cifar100 
 
# python3 train.py --model DarkNet19 --channel 64 --epochs 200 --eval_term 5 --dataset cifar100 --batch_size 100 
# python3 train.py --model ResNet18 --channel 64 --epochs 200 --eval_term 5 --dataset cifar100 --batch_size 100
# python3 train.py --model VGG19 --channel 64 --epochs 200 --eval_term 5 --dataset cifar100 --batch_size 100
# python3 train.py --model GoogleNet22 --channel 64 --epochs 200 --eval_term 5 --dataset cifar100 --batch_size 100

# python3 train.py --model DarkNet19 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model ResNet18 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model VGG19 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
# python3 train.py --model GoogleNet22 --channel 64 --epochs 500 --eval_term 5 --dataset imagenet --batch_size 100
start_background_task() {
    nohup python3 "$1.py --model $2 --channel 64 --eval_term 5 --dataset $3 --optimizer SGD > $2.log" 2>&1 &
}

# 대기 함수
wait_for_completion() {
    while pgrep -f "python3 $1.py --model $2 --channel 64 --eval_term 5 --dataset $3 --optimizer SGD" > /dev/null; do
        sleep 1
    done
}

task() {
    start_background_task "$1 $2 $3 $4"
    wait_for_completion "$1 $2 $3 $4"
}

task train SGD mnist