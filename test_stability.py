from types import SimpleNamespace
import data_loader
import models
import os
import constants
import torch
from tqdm import tqdm  # optional progress bar
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

args = SimpleNamespace(dataset="CIFAR10", batch_size=32, num_workers=4, model="resnet50", image_size=32, num_class=10, num_per_class=10)

model = getattr(models, args.model)(pretrained=False)
path = os.path.join(constants.PRETRAINED_MODELS, ('%s/%s.pt' % (args.dataset, args.model)))
model.load_state_dict(torch.load(path))

TOTAL_CLASS_NUM, train_loader, test_loader, seed_loader = data_loader.get_loader(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# ---------- Helper function ----------

class ImageSeedsDatasetLazy(Dataset):
    def __init__(self, meta_list):
        self.meta = list(meta_list)
        self.transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        path, adv_label, orig_label  = self.meta[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), adv_label, orig_label


def load_fuzzed_inputs(data_dir, ):

    samples = []
    for root, _, files in os.walk(data_dir):
        for filename in files:
            original_label = int(root.split("/")[-1])
            parts = filename.split('_')
        
            try:
                adversarial_label = int(parts[4].split(".")[0])
            except:
                adversarial_label = original_label
        
            img_path = os.path.join(root, filename)
            samples.append((img_path, adversarial_label, original_label))
    
    fuzzed_dataset = ImageSeedsDatasetLazy(samples)
    return fuzzed_dataset
        

def evaluate(model, loader, device, preservation=True):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, adv_labels, orig_labels in tqdm(loader, desc="Evaluating", leave=False):
            images, adv_labels, orig_labels = images.to(device), adv_labels.to(device), orig_labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += adv_labels.size(0)
            if preservation:
                correct += (predicted == adv_labels).sum().item()
            else:
                correct += (predicted == orig_labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, correct

def collate_fn(batch):
        if len(batch[0]) == 3:
            images, adversarial_labels, labels = zip(*batch)
            adversarial_labels = torch.tensor(adversarial_labels)
        else:
            images, labels = zip(*batch)
            adversarial_labels = None
            
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, adversarial_labels, labels

fuzzed_dataset = load_fuzzed_inputs("data/output/Coverage/Fuzzer/CIFAR10-resnet50-NLC/image/aes")
fuzzed_dataset_loader = DataLoader(fuzzed_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

# ---------- Measure accuracies ----------
# train_acc = evaluate(model, train_loader, device)
# test_acc = evaluate(model, test_loader, device)
fuzzed_acc, count = evaluate(model, fuzzed_dataset_loader, device, preservation=True)

# print(f"Train Accuracy: {train_acc:.2f}%")
# print(f"Test Accuracy:  {test_acc:.2f}%")
print(f"Fuzzed Accuracy:  {fuzzed_acc:.2f}%, Count: {count}")
