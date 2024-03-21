log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 "$1".py --model "$2" --official "$3" --dataset "$4" --gpus "$5" --epochs "$6" --channel 64 --batch_size 100 > "$7/$2_$3_$4.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 $1.py --model $2 --official $3 --dataset $4 --gpus $5 --epochs $6 --channel 64 --batch_size 100" > /dev/null; do
        sleep 5
    done
}

task() {
    start_background_task "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" 
    wait_for_completion "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" 
}

log_dir="./log"
log_directory_check "$log_dir"

task model_train ResNet18 True cifar10 0 300 "$log_dir"
task model_train ResNet18 False cifar10 0 300 "$log_dir"
task model_train DarkNet19 False cifar10 0 300 "$log_dir"
task model_train GoogleNet22 True cifar10 0 300 "$log_dir"
task model_train GoogleNet22 False cifar10 0 300 "$log_dir"
