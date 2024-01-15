import psutil, subprocess, cpuinfo, os, time

COLOR_CODES = {'red': '\033[91m', 'orange': '\033[38;2;255;200;100m', 'blue': '\033[94m', 'green': '\033[92m', '': '\033[0m'}

def get_cpu_info():
    # Gets temperatures for each CPU, if available
    temps = psutil.sensors_temperatures()
    cpu_temps = []
    for hwmon in temps.values():
        for sensor in hwmon:
            if 'Package' in sensor.label:
                cpu_temps.append(sensor.current)

    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)

    return [cpu_usage, cpu_temps]

def get_cpu_hw_info():
    cpu_info = cpuinfo.get_cpu_info()
    return [cpu_info['brand_raw'], cpu_info['count']]

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        infos = [info.split(',') for info in result.stdout.strip().split('\n')]
        gpu_info = [[index, device, f'{temp[1:]}°C', f'{float(pow_use):.0f}/{float(pow_lim):.0f}W', f'{float(mem_use)/1024:.1f}/{float(mem_lim)/1024:.1f}Gb', f'{volt[1:]}%'] for index, device, temp, pow_use, pow_lim, mem_use, mem_lim, volt in infos]
        return [sorted(gpu_info, key=lambda x: x[0])]
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def get_memory_info():
    return [psutil.virtual_memory(), psutil.swap_memory()]

def print_colored(text, color, end='\n'):
    color_code = COLOR_CODES.get(color, '')
    print(f"{color_code}{text}\033[0m", end=end)

def print_cpu(cpu_info, cpu_usage, cpu_temps, f):
    l = (f-1)//8
    
    title = f'‖ {len(cpu_temps)} CPUs ‖ {len(cpu_usage)} cores ‖ '
    width = ((len(title)+6*len(cpu_temps) + 2*len(cpu_temps)))
    pad = (f-width)//2
    print_colored('=' * (pad) + title, 'blue', end='')

    for i, t in enumerate(cpu_temps, start=1):
        print_colored(str(t)+ '°C', 'red' if t>=70 else 'orange' if t>=60 else 'blue', end='')
        if i!=len(cpu_temps):
            print_colored(', ', 'blue', end='')
        else:
            print_colored(' ‖'+'='*(f-width-pad), 'blue')

    print_colored('‖', 'blue', end='')
    print_colored(f'{cpu_info[0]:^{f-2}}', 'green', end='')
    print_colored('‖', 'blue')

    w = 5
    l_pad = (l - 4 - w)//2
    r_pad = l - 4 - w - l_pad - 1
    for i, usage in enumerate(cpu_usage, start=1):
        print_colored('‖', 'blue', end='')
        print(f'{i: >2}|', end='')
        print_colored(f'{" "*l_pad}{usage:>{w}.1f}%{" "*r_pad}', 'red' if usage >= 70 else 'orange' if usage >= 40 else '', end='')
        # print_colored(f'{f"{usage:.1f}%":>{l-4}}', 'red' if usage >= 70 else 'orange' if usage >= 40 else '', end='')
        if i % 8 == 0 or i == len(cpu_usage):
            print_colored('‖', 'blue')

def print_gpu(gpu_info, f):
    l = (f-4)//5
    s = max(25, l)
    d = abs(s-l)//3

    title = f'‖ {len(gpu_info)} GPUs ‖'
    t = (f-len(title))//2
    print_colored('=' * t + title + '=' * (f - t - len(title)), 'blue')
    
    print_colored('‖', 'blue', end='')
    print_colored(f'{" "*2}|{"Device":^{s}}|{"Temp":^{l-d-6}}|{"Power":^{l-d+3}}|{"Memory":^{l-d+3}}|{"Volt":^{f-9-(l-d)*3-s}}', 'green', end='')
    print_colored('‖', 'blue')
    for infos in gpu_info:
        print_colored('‖', 'blue', end='')
        for id, info, length, w in zip(['index', 'device', 'temp', 'power', 'memory', 'volt'], infos, [2, s, l-d-6, l-d+3, l-d+3, f-9-(l-d)*3-s], [0, 25, 4, 8, 11, 4]):
            if id == 'index':
                print(f'{info:>{length}}', end='')
            else:
                l_pad = (length - w)//2
                r_pad = length - w - l_pad
                print('|', end='')
                if id == 'device':
                    print(f'{" "*l_pad}{info:<{w}}{" "*r_pad}', end='')
                elif id == 'temp':
                    print_colored(f'{" "*l_pad}{info:>{w}}{" "*r_pad}', 'red' if int(info[:-2]) >= 60 else 'orange' if int(info[:-2]) >= 50 else '', end='')
                else:
                    print(f'{" "*l_pad}{info:>{w}}{" "*r_pad}', end='')
        print_colored('‖', 'blue')
def print_mem(memory, swap, f):
    l = f//2

    title = f'‖ RAM ‖'
    t = (f-len(title))//2
    print_colored('=' * t + title + '=' * (f - t - len(title)), 'blue')

    print_colored('‖', 'blue', end='')
    print_colored(f'{"Memory":^{l-1}}|{"Swap":^{f-l-2}}', 'green', end='')
    print_colored('‖', 'blue')

    print_colored('‖', 'blue', end='')
    print(f'{f"{memory.used/1024**3:.1f}/{memory.total/1024**3:.1f}Gb":^{l-1}}|{f"{swap.used/1024**3:.1f}/{swap.total/1024**3:.1f}Gb":^{l-1}}', end='')
    print_colored('‖', 'blue')

def print_system_status():
    MIN = 10
    cpu_info = get_cpu_hw_info()

    l = max(MIN, 10)
    f = l * 8 + 1

    while True:
        # info
        a = get_cpu_info()
        b = get_gpu_info()
        c = get_memory_info()
        os.system('clear')
        print_cpu(cpu_info, *a, f)
        print_gpu(*b, f)
        print_mem(*c, f)
        print_colored('='*f, 'blue')
        # time.sleep(1)
        
if __name__ == "__main__":
    print_system_status()
