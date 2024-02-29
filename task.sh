log_directory_check(){
    log_dir=$1
    if [ -d "$log_dir" ]; then
        :
    else
        mkdir -p "$log_dir"
    fi
}

start_background_task() {
    nohup python3 "$1".py --network "$2" --model "$3" --dataset "$4" --gpus "$5" --n_tasks "$6" --n_memories "$7" --epochs "$8" --channel 64 --batch_size 100 > "$9/$2_$3_$4.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 $1.py --network $2 --model $3 --dataset $4 --gpus $5 --n_tasks $6 --n_memories $7 --epochs $8 --channel 64 --batch_size 100" > /dev/null; do
        sleep 5
    done
}

task() {
    start_background_task "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" 
    wait_for_completion "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" 
}

log_dir="./log"
log_directory_check "$log_dir"
# task network_train GEM ResNet18 mnist 0 5 256 10 "$log_dir"
# task network_train GEM ResNet18 cifar10 0 5 256 10 "$log_dir"
task network_train GEM ResNet18 cifar100 0 20 256 30 "$log_dir"
task network_train single ResNet18 cifar100 0 20 256 10 "$log_dir"