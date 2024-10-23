import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import datetime
import os
from  util.logger_util import setup_logger
from util.my_dataSet import MyDataset
import json
from torch.optim.lr_scheduler import CosineAnnealingLR,ReduceLROnPlateau


def main(k_fold):
    # k_fold = 4
    json_path = r'../util/folds_info.json'
    root_dir =  r'../dataset/LIMUC/train_and_validation_sets'

    model_name = 'swinv2_base_window8_256.ms_in1k'
    pretrained_file = model_name + ".bin"
    patience = 10
    log_file = os.path.join(os.getcwd(), "log.txt")
    logger = setup_logger(log_file)


    entries = os.listdir(root_dir)
    directories = [entry for entry in entries if os.path.isdir(os.path.join(root_dir, entry))]
    print(directories)


    with open(json_path, 'r') as file:
        folds_info = json.load(file)

    data_fold = folds_info[k_fold]
    train_data = data_fold['train_images']
    val_data = data_fold['val_images']

    logger.info(f'------------ {k_fold} fold ------------------')
    print(f'------------ {k_fold} fold ------------------')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 新增垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 新增颜色抖
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MyDataset(train_data, root_dir, transform=transform_train)
    val_dataset = MyDataset(val_data, root_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = timm.create_model(model_name, pretrained=True,
        pretrained_cfg_overlay=dict(file=pretrained_file), num_classes=len(directories))
    model.to(device)

    num_epochs = 200

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    best_accuracy = 0.0
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # 重新计算验证损失
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_dataset)
        accuracy = total_correct / len(val_dataset)

        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S:%f')
        print(time_str)
        msg = (f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.4f}')

        print(msg)
        logger.info(msg)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            counter = 0
            torch.save(model.state_dict(), f'./output/best_model_{k_fold}_{epoch + 1}.pth')
            logger.info(f'Best accuracy model: {epoch + 1}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement.')
                logger.info(f'Early stopping after {patience} epochs without improvement.')
                break

    print(f'Best Val Loss: {best_val_loss:.4f}')

    print(f'Best validation accuracy: {best_accuracy:.4f}')

if __name__ == '__main__':
    for i in range(5):
        main(i)
