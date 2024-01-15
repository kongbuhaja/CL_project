#!/bin/bash

# /proc/stat에서 CPU 사용 정보 읽기
while true; do
  while read -r line; do
      # CPU 라인만 파싱
      case "$line" in
          cpu[0-9]*)
              # CPU 번호와 사용량 추출
              cpu=$(echo $line | awk '{print $1}')
              user=$(echo $line | awk '{print $2}')
              nice=$(echo $line | awk '{print $3}')
              system=$(echo $line | awk '{print $4}')
              idle=$(echo $line | awk '{print $5}')

              # CPU 사용률 계산
              total=$((user + nice + system))
              total_idle=$((total + idle))
              cpu_usage=$(awk "BEGIN {print ($total / $total_idle) * 100}")

              # 결과 출력
              echo "CPU $cpu: Usage: $cpu_usage%"
              ;;
      esac
  done < /proc/stat
  clear
done