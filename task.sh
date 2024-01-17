log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 "$1".py --model "$2" --channel 64 --eval_term 5 --dataset "$3" > "$4/$2.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 $1.py --model $2 --channel 64 --eval_term 5 --dataset $3 " > /dev/null; do
        sleep 5
    done
}

task() {
    start_background_task "$1" "$2" "$3" "$4"
    # wait_for_completion "$1" "$2" "$3"
}

log_dir="./log"
log_directory_check "$log_dir"
task train VGG19 imagenet "$log_dir"
