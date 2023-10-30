import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='base model')

    parser.add_argument('--model', dest='model', type=str, default='DarkNet19', help='model to train')
    parser.add_argument('--loss' , dest='loss', type=str, default='CE', help='function to train model')
    parser.add_argument('--load', dest='load', type=bool, default=False, help='whether to load model')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1000, help='batch_size for training or inference')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='learning rate for training')

    parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='dataset for training')
    parser.add_argument('--image_size', dest='image_size', type=str, default='28x28', help='dataset for training')

    args = parser.parse_args()
    print(f'dataset: {args.dataset}')
    print(f'model: {args.model}')
    print(f'loss: {args.loss}')

    return args