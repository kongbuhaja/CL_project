start_background_task() {
    nohup python3 "$1".py --model "$2" --channel 64 --eval_term 5 --dataset "$3" --optimizer SGD > "$2.log" 2>&1 &
}

wait_for_completion() {
    while pgrep -f "python3 $1.py --model $2 --channel 64 --eval_term 5 --dataset $3 --optimizer SGD" > /dev/null; do
        sleep 5
    done
}

task() {
    start_background_task "$1" "$2" "$3"
    wait_for_completion "$1" "$2" "$3"
}

task val ResNet18 mnist