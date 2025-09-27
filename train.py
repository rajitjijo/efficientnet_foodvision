import torch
import torch.nn as nn
import torchvision
from datasetup import create_dataloaders
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import engine
from datetime import datetime
import os


train_dir = "data/train"
test_dir = "data/test"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
model_name = "efficientnet"
experiment_name = "minifood"

def create_writer(experiment_name, model_name, extra=None):

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    return SummaryWriter(log_dir=log_dir)
        

if __name__ == "__main__":
    
    #Setting up pretrained weights
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    #Get Transforms from weights
    transforms = weights.transforms()
    train_loader, test_loader, class_names = create_dataloaders(train_dir, test_dir, transforms, batch_size=batch_size)

    model = torchvision.models.efficientnet_b0(weights=weights)
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(num_features, len(class_names))     # type: ignore
    
    # summary_str = summary(
    #         model,
    #         input_size=(10,3,224,224),
    #         col_names=["input_size","output_size","num_params", "trainable"],
    #         row_settings=["var_names"],
    #         col_width=20,
    #         verbose=0
    #     )

    # with open("model_summary.txt", "w", encoding="utf-8") as f:
    #     f.write(str(summary_str))

    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    #To Track experiments
    writer = create_writer(experiment_name, model_name)

    results = engine.train(model, train_loader,test_loader, optimzer, criterion, num_epochs, writer, device)

    


