import csv
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from easyrobust.benchmarks import *
from imagenet_x import error_ratio, get_factor_accuracies
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


@torch.no_grad()
def get_correct_prob(data_loader, model, device):
    model.eval()

    total_correct_class_probs = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(data_loader):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)

        probs = F.softmax(logits, dim=1)
        correct_class_probs = probs[torch.arange(images.size(0)), target]

        _, pred = torch.max(logits, 1)
        total_correct += (pred == target).sum().item()

        total_correct_class_probs += correct_class_probs.sum().item()
        total_samples += images.size(0)

    avg_correct_class_prob = total_correct_class_probs / total_samples
    top1_accuracy = total_correct / total_samples

    return {
        "correct_class_prob": avg_correct_class_prob,
        "top1_accuracy": top1_accuracy,
    }


@torch.no_grad()
def get_accuracy_imagenet_real(data_loader, model, device,
                               real_labels_imagenet):
    model.eval()

    for batch in tqdm(data_loader):
        images, _ = batch
        images = images.to(device, non_blocking=True)

        logits = model(images)

        real_labels_imagenet.add_result(logits)

    accuracy_metrics = real_labels_imagenet.get_accuracy(k=1)

    return {"acc_real": accuracy_metrics}


@torch.no_grad()
def run_imagenetx(data_loader, model, device, model_name):
    model.eval()

    total_samples = 0
    max_probs_list = []
    max_indices_list = []
    file_names_list = []

    date = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    folder = os.path.join(os.environ["HOME"],
                          f"outputs/imagenetx/{date}_{model_name}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_csv_path = os.path.join(folder, "preds.csv")

    with open(output_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["file_name", "predicted_class", "predicted_probability"])

        for batch in tqdm(data_loader):
            images, target, file_names = batch

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            logits = model(images)

            probs = F.softmax(logits, dim=1)
            max_probs, max_indices = torch.max(probs, dim=1)

            max_probs_list.append(max_probs.detach().cpu().numpy())
            max_indices_list.append(max_indices.detach().cpu().numpy())
            file_names_list.append(file_names)

            for file_name, pred_class, pred_prob in zip(
                    file_names, max_indices, max_probs):
                csv_writer.writerow(
                    [file_name, pred_class.item(),
                     pred_prob.item()])

            total_samples += images.shape[0]

    print("General inference finished successfully!")
    factor_accs = get_factor_accuracies(
        os.path.join(os.environ["HOME"],
                     f"outputs/imagenetx/{date}_{model_name}"))
    return error_ratio(factor_accs)


@torch.no_grad()
def run_imagenethard(data_loader, model, device):
    print("Calculating validation accuracy!")
    model.eval()

    correct_ones = 0

    for batch in tqdm(data_loader):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        model_output = model(images)

        preds = model_output.data.max(1)[1]
        correct_ones += (preds[:, None] == target).any(1).sum().item()

    print("Validation inference finished successfully!")

    accuracy = 100 * correct_ones / len(data_loader.dataset)
    print(len(data_loader.dataset))
    print(f"Validation accuracy: {accuracy:.2f}%")
    return {"accuracy": accuracy}


def get_pug_imagenet(model, root_folder, transform_val=None):
    with open(os.path.join(root_folder, "class_to_imagenet_idx.json")) as f:
        labels = json.load(f)
    labels = dict(sorted(labels.items()))
    inversed_dict = {}
    counter = 0
    for k, v in labels.items():
        for val in v:
            inversed_dict[int(val)] = counter
        counter = counter + 1

    if transform_val is None:
        tr_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        transform_val = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            tr_normalize,
        ])

    dataset_names = [
        "Worlds",
        "Camera_Pitch",
        "Camera_Yaw",
        "Camera_Roll",
        "Object_Pitch",
        "Object_Yaw",
        "Object_Roll",
        "Object_Scale",
        "Object_Texture",
        "Scene_Light",
    ]

    results = {}

    for dataset_name in dataset_names:
        dataset_path = os.path.join(root_folder, dataset_name)
        dataset = datasets.ImageFolder(dataset_path, transform=transform_val)
        dataloader = DataLoader(dataset,
                                batch_size=256,
                                shuffle=False,
                                num_workers=10,
                                drop_last=False)

        print(f"Running inference on {dataset_name}.")

        nb_corrects = 0.0
        for images, labels in tqdm(dataloader):
            images = images.cuda()
            labels = labels.cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(images).softmax(dim=-1)
                pred = torch.argmax(output, dim=1)
                for p in range(pred.size(0)):
                    if pred[p].item() in inversed_dict.keys():
                        pred[p] = inversed_dict[pred[p].item()]
                    else:
                        pred[p] = 999
                nb_corrects += sum((pred == labels).float())

        accuracy = (nb_corrects / len(dataset)) * 100.0
        results[dataset_name] = accuracy.item()

    return results


@torch.no_grad()
def run_robustness(model, root, test_transform=None):
    model.eval()
    acc_val = evaluate_imagenet_val(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-val"),
        test_transform=test_transform,
    )

    acc_a = evaluate_imagenet_a(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-a"),
        test_transform=test_transform,
    )

    acc_r = evaluate_imagenet_r(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-r"),
        test_transform=test_transform,
    )

    acc_sketch = evaluate_imagenet_sketch(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-sketch"),
        test_transform=test_transform,
    )

    acc_v2 = evaluate_imagenet_v2(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenetv2"),
        test_transform=test_transform,
    )

    acc_style = evaluate_stylized_imagenet(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-style"),
        test_transform=test_transform,
    )
    acc_c, _ = evaluate_imagenet_c(
        model=model,
        data_dir=os.path.join(root, "benchmarks/data/imagenet-c"),
        test_transform=test_transform,
    )

    return {
        "acc_a": acc_a,
        "acc_r": acc_r,
        "acc_sketch": acc_sketch,
        "acc_v2": acc_v2,
        "acc_style": acc_style,
        "acc_val": acc_val,
        "acc_c": acc_c,
    }
