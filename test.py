 
switch = 0
import torchsummary
Device = "cpu"

if switch == 0:
    import torch, tqdm
    from torch.utils.data import DataLoader
    

    from model.utils import *
    from data.utils import *
    from utils import *
    from val import eval

    args = args_parse()
    args_show(args)
    env_set(args.gpus)

    # Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    train_dataset, val_dataset = load_dataset(args.dataset, args.image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.cpus)

    model, start_epoch, best_recall, recalls, save_path = load_model(args.dataset, args.optimizer, args.model, args.channel, 
                                                                    len(train_dataset.unique_labels), args.eval_term,
                                                                    load=args.load)
    model.to(Device)
    torchsummary.summary(model, (256,256,3), device=Device)
    # print(model.layers)


elif switch == 1:
    import torchvision

    # model = torchvision.models.vgg16_bn()
    # model = torchvision.models.resnet18()
    model = torchvision.models.googlenet()
    model.to(Device)
    torchsummary.summary(model, (3,256,256), device=Device)
    # print(model)
