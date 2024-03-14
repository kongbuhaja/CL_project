# CL_project  
  
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
- [ ] Implement Contiual learning model and metric  
- [ ] Make Comparable graph  
- [ ] Suggest new continual learning model

## Performance  
resize image(256x256) with pad.  
official: get model from torchvision  

### Normal data
| Model | official | MNIST | CIFAR10 | CIFAR100 | IMAGENET |    
| ------------- | :---: | :------: | :------: | :------: | :------: |
| **ResNet18**  | O | 98.8% | - | 61.0% | 53.8% |
| **ResNet18**  | X | 98.5% | - | 62.1% | 56.6% |
| **GoogleNet22**  | O | 98.9% | - | 66.4% | - |
| **GoogleNet22**  | X | 98.7% | - | 67.5% | 49.7% |
| **DarkNet19**  | X | 98.4% | - | 65.2% | 55.8% |
| ~~VGG16~~ | O | - | - | - | - |
| ~~VGG16~~ | X | - | - | - | - |

### Continuum data
| Model | official | MNIST | CIFAR10 | CIFAR100 | IMAGENET |    
| ------------- | :---: | :------: | :------: | :------: | :------: |
| **ResNet18**  | O | - | - | - | - |
| **ResNet18**  | X | - | - | - | - |
| **GoogleNet22**  | O | - | - | - | - |
| **GoogleNet22**  | X | - | - | - | - |
| **DarkNet19**  | X | - | - | - | - |
| ~~VGG16~~ | O | - | - | - | - |
| ~~VGG16~~ | X | - | - | - | - |
