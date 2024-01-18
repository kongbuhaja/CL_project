log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 "$1".py --model "$2" --gpus $3 --channel 64 --batch_size 150 --eval_term 5 --dataset "$4" --load "$5" > "$6/$2.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 $1.py --model $2 --gpus $3 --channel 64 --batch_size 150  --eval_term 5 --dataset $4 --load $5" > /dev/null; do
        sleep 5
    done
}

task() {
    start_background_task "$1" "$2" "$3" "$4" "$5" "$6"
    # wait_for_completion "$1" "$2" "$3" "$4" "$5"
}

log_dir="./log"
log_directory_check "$log_dir"
task train DarkNet19 0 imagenet False "$log_dir"
task train ResNet18 1 imagenet False "$log_dir"
task train VGG16 2 imagenet False "$log_dir"
# task train VGG19 2 imagenet False "$log_dir"
task train GoogleNet22 3 imagenet False "$log_dir"
