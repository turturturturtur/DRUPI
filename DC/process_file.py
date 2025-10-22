import os
import numpy as np
import torch


def process_file(file_path):
    # Read from pytorch pt
    result = torch.load(file_path)
    data, acc = result["data"], result["accs_all_exps"]
    # average every 20 data points
    acc = acc["ConvNet"]
    acc = np.array(acc)
    acc = acc.reshape(-1, 20)
    acc = np.mean(acc, axis=1)
    target_exp = np.argmax(acc)
    data = data[target_exp]
    image, label = data[0], data[1]
    return image, label


def save_file(dataset, ipc, method, image, label):
    save_path = f"/home/zhanglf/FD/distilled_data/{method}/{dataset}/IPC{ipc}"
    os.makedirs(save_path, exist_ok=True)
    torch.save(image, os.path.join(save_path, "images_best.pt"))
    torch.save(label, os.path.join(save_path, "labels_best.pt"))


if __name__ == "__main__":
    directory = "/home/zhanglf/FD/distilled_data"
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):  
            file_path = os.path.join(directory, filename) 
            args = file_path.split("/")[-1].split("_")
            method, dataset, ipc = args[1], args[2], args[-1].split(".")[0][:-3]
            image, label = process_file(file_path)
            save_file(dataset, ipc, method, image, label)
