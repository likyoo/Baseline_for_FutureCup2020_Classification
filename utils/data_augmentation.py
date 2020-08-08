import torchvision.transforms as transforms
from conf import settings

mean = settings.IMAGENET_TRAIN_MEAN
std = settings.IMAGENET_TRAIN_STD

transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])