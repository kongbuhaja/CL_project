import os, argparse, time
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--file', dest='file', type=str, default='train', help='which file do you want to read')
parser.add_argument('--lines', dest='lines', type=int, default=300, help='number of reading lines')
parser.add_argument('--times', dest='times', type=float, default=1., help='number of reading lines')
args = parser.parse_args()
args.file += '.log'

def head(args):
    with open(args.file, 'r') as f:
        lines = f.readlines()[:30]

    l = 0 if 'nohup' in lines[0] else 1
    while(l<len(lines)):
        print(lines[l].strip())
        l += 1
        try:
            if 'Create' in lines[l] or 'Success' in lines[l]:
                break
        except:
            break
    
def tail(args):
    with open(args.file, 'r') as f:
        lines = f.readlines()[-args.lines:]

    l = 0
    while(l<len(lines)):
        print(lines[l].strip(), end='\r')
        if '100%' in lines[l]:
            print()
            l += 1
        l += 1

while True:
    os.system('clear')
    print(f'System Time | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    head(args)
    tail(args)
    time.sleep(args.times)
