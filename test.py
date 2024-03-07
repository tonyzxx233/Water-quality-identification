# 验证模型准确率
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
from model_train import SELFMODEL

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

data_path = "./image/water_data_split"  # todo 修改为数据集根目录
model_path = "./checkpoints/resnet50d_pretrained_224/resnet50d_10epochs_accuracy0.99970_weights.pth"  # todo 模型地址
model_name = 'resnet50d'  # todo 模型名称
# model_name = 'efficientnet_b3a'
# model_name = 'vit_tiny_patch16_224'
img_size = 224  # todo 数据集训练时输入模型的大小
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    # 'model': 'vit_tiny_patch16_224',  # 选择预训练模型
    # 'model': 'efficientnet_b3a',  # 选择预训练模型
    'model': model_name,  # 选择预训练模型
    "img_size": img_size,  # 图片输入大小
    "test_dir": osp.join(data_path, "test"),  # todo 测试集子目录
    'device': device,  # 设备
    'batch_size': 4,  # 批次大小
    'num_workers': 0,  # 进程
    "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
}


def test(val_loader, model, params, class_names):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            # loss = criterion(output, target.long())  # 计算损失
            # print(output)
            target_numpy = target.cpu().numpy()
            y_pred = torch.softmax(output, dim=1)
            y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            test_real_labels.extend(target_numpy)
            test_pre_labels.extend(y_pred)
            # print(target_numpy)
            # print(y_pred)
            f1_macro = calculate_f1_macro(output, target)  # 计算f1分数
            recall_macro = calculate_recall_macro(output, target)  # 计算recall分数
            acc = accuracy(output, target)  # 计算acc
            # metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('F1', f1_macro)
            metric_monitor.update("Recall", recall_macro)
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "mode: {epoch}.  {metric_monitor}".format(
                    epoch="test",
                    metric_monitor=metric_monitor)
            )
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    # print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    # print()
    heat_maps_float = heat_maps / heat_maps_sum
    # print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="record/heatmap_{}.png".format(model_name))
    # 加上模型名称

    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['F1']["avg"], \
           metric_monitor.metrics['Recall']["avg"]


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


if __name__ == '__main__':
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    # train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    # valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    # print(valid_dataset)
    test_dataset = datasets.ImageFolder(params["test_dir"], valid_transforms)
    class_names = test_dataset.classes
    print(class_names)
    # valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    test_loader = DataLoader(  # 按照批次加载训练集
        test_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )

    # 加载模型
    model = SELFMODEL(model_name=params['model'], out_features=params['num_classes'],
                      pretrained=False)  # 加载模型结构，加载模型结构过程中pretrained设置为False即可。
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    # 指标上的测试结果包含三个方面，分别是acc f1 和 recall, 除此之外，应该还有相应的热力图输出，整体会比较好看一些。
    acc, f1, recall = test(test_loader, model, params, class_names)
    print("测试结果：")
    print(f"acc: {acc}, F1: {f1}, recall: {recall}")
    print("测试完成，heatmap保存在{}下".format("record"))