import torch
from torchstat import stat
from model_train import SELFMODEL

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model_name = "resnet50d" # todo 模型名称
num_classes = 4 # todo 类别数目
model_path = "checkpoints/resnet50d_pretrained_224/resnet50d_9epochs_accuracy0.99960_weights.pth"  # todo 模型地址
model = SELFMODEL(model_name=model_name, out_features=num_classes, pretrained=False)
weights = torch.load(model_path)
model.load_state_dict(weights)
model.eval()
stat(model, (3, 224, 224)) # 后面的224表示模型的输入大小