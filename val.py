import torch, cv2, tqdm
import numpy as np
from torch.utils.data import DataLoader

from model.utils import load_model
from data.utils import *
from utils import *


def main():
    args = args_parse()
    args.batch_size = 1
    args_show(args, train=False)
    env_set(args.gpus)

    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False)

    model, start_epoch, best_recall, recalls, save_path = load_model(args.dataset, args.optimizer, args.model, args.channel, 
                                                                    len(train_dataset.unique_labels), args.image_size, args.eval_term,
                                                                    load=args.load)
    model.to(Device)

    loss_fn = loss_function(args.loss, len(train_dataset.unique_labels))
    
    test_x_data = []
    test_p_data = []
    test_y_data = []

    model.eval()
    with torch.no_grad():
        positive = 0
        val_loss = 0.
        val_tqdm = tqdm.tqdm(val_dataloader, total=len(val_dataloader), ncols=121, desc=f'Validation', ascii=' =', colour='blue')
        for iter, (x_data, y_data) in enumerate(val_tqdm):
            pred = model(x_data.to(Device))
            loss = loss_fn(pred, y_data[..., 0].to(Device))

            val_loss += loss.item()

            pred_label = torch.argmax(pred, -1).to('cpu')
            positive += sum(pred_label == y_data[..., 0])
            
            recall = positive/((iter+1)*args.batch_size)
            
            if len(test_x_data) == iter and len(test_x_data) < 9:
                test_x_data += [x_data[0].to('cpu').numpy()]
                test_p_data += [pred_label[0].to('cpu').numpy()]
                test_y_data += [y_data[0][0].to('cpu').numpy()]
            
            val_tqdm.set_postfix_str(f'| recall: {recall:.3f}, val_loss: {val_loss/(iter+1):.4f}')
    
    row = None
    output = None

    for i, (x, p, y) in enumerate(zip(test_x_data, test_p_data, test_y_data)):
        x = (x*255).astype(np.uint8)
        if x.shape[-1] == 1:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
        x = cv2.resize(x, [100, 100])
        x = cv2.putText(x, str(p), (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
        x = cv2.putText(x, str(y), (82,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        if row is None:
            row = x
        else:
            row = np.concatenate([row, x], 1)
            if i%3 == 2:
                if output is None:
                    output = row
                else:
                    output = np.concatenate([output, row], 0)
                row = None

    output_dir = f'output/{args.dataset}/{args.model}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(f'{output_dir}/output.jpg', output)

def eval(model, val_dataloader, loss_fn, Device):
    model.eval()
    with torch.no_grad():
        positive = 0
        val_loss = 0.
        val_tqdm = tqdm.tqdm(val_dataloader, total=len(val_dataloader), ncols=121, desc=f'Validation', ascii=' =', colour='blue')
        for iter, (x_data, y_data) in enumerate(val_tqdm):
            pred = model(x_data.to(Device))
            loss = loss_fn(pred, y_data[..., 0].to(Device))

            val_loss += loss.item()
            
            pred_label = torch.argmax(pred, -1).to('cpu')
            positive += sum(pred_label == y_data[..., 0])
            recall = positive/((iter+1)*val_dataloader.batch_size)
            val_tqdm.set_postfix_str(f'| recall: {recall:.3f}, val_loss: {val_loss/(iter+1):.4f}')
    return recall.numpy(), val_loss/(iter+1)

if __name__ == '__main__':
    main()