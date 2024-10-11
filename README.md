# CL_project  
- Pytorch based
## Processes  
### Data  
- [x] Download data  
- [x] Data structure  
- [x] Augmentation  
  
### Model
- [x] Resnet18  
- [x] Darknet19  
- [x] Googlenet22  
- [x] VGG19  
  
### Optimizer
- [x] SGD  
- [x] Momentum  
- [x] Adam  
- [x] Adadelta 
- [x] Adagrad  
- [x] AdamW  
- [x] NAdam  
- [x] RAdam 
- [x] RMSprop  
- [x] Rprop

### LR_scheduler:
- [x] static
- [x] linear
- [x] cosine_annealing

### Task  
- [x] Train Based Model  
- [x] Evaluate Based Model
- [x] Implement Contiual learning model and metric  
- [x] Make Comparable graph  
- [ ] Suggest new continual learning model

## Performance  
Resize image(256x256) with pad.  
Official: get model from torchvision  

### Normal data
| Model | official | MNIST | CIFAR10 | CIFAR100 | IMAGENET |    
| ------------- | :---: | :------: | :------: | :------: | :------: |
| **ResNet18**  | O | 98.8% | 90.3% | 61.0% | 53.8% |
| **ResNet18**  | X | 98.5% | 89.5% | 62.1% | 56.6% |
| **GoogleNet22**  | O | 98.9% | 89.5% | 66.4% | 63.0% |
| **GoogleNet22**  | X | 98.7% | 85.2% | 67.5% | 49.7% |
| **DarkNet19**  | X | 98.4% | 90.2% | 65.2% | 55.8% |
| ~~VGG16~~ | O | - | - | - | - |
| ~~VGG16~~ | X | - | - | - | - |

### Continuum data
| Model | official | MNIST | CIFAR10 | CIFAR100 | IMAGENET |    
| ------------- | :---: | :------: | :------: | :------: | :------: |
| **ResNet18**  | O | - | - | - | - |
| **ResNet18**  | X | 55.8% | - | - | - |
| **GoogleNet22**  | O | - | - | - | - |
| **GoogleNet22**  | X | - | - | - | - |
| **DarkNet19**  | X | - | - | - | - |
| ~~VGG16~~ | O | - | - | - | - |
| ~~VGG16~~ | X | - | - | - | - |
