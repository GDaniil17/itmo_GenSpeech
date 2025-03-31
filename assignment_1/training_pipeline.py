from tqdm import tqdm
import os
import time
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from melbanks import LogMelFilterBanks


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, root, subset=None):
        super().__init__(root, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath, "r") as f:
                return [os.path.join(self._path, line.strip()) for line in f]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            # Исключаем файлы валидации и тестирования
            val_list = set(load_list("validation_list.txt"))
            test_list = set(load_list("testing_list.txt"))
            excludes = val_list.union(test_list)
            self._walker = [w for w in self._walker if w not in excludes]


class YesNoDataset(torch.utils.data.Dataset):
    _cache = {}

    def __init__(self, root, subset, transform=None):
        self.dataset = SubsetSC(root, subset=subset)
        self.transform = transform
        key = f"{root}_{subset}"
        if key in YesNoDataset._cache:
            self.yesno_samples = YesNoDataset._cache[key]
        else:
            self.yesno_samples = []
            for waveform, sr, label, _, _ in tqdm(self.dataset, desc="Фильтрация образцов yes/no"):
                if label in ["yes", "no"]:
                    self.yesno_samples.append((waveform, sr, label))
            YesNoDataset._cache[key] = self.yesno_samples
        self.label_map = {"no": 0, "yes": 1}

    def __len__(self):
        return len(self.yesno_samples)

    def __getitem__(self, idx):
        waveform, sr, label = self.yesno_samples[idx]
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0, keepdim=True)
        if self.transform is not None:
            waveform = self.transform(waveform)
        target = self.label_map[label]
        return waveform, target


class SimpleCNN(nn.Module):
    def __init__(self, n_mels=80, groups=1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, 16, kernel_size=3, padding=1, groups=groups)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1, groups=groups)
        self.pool = nn.MaxPool1d(2)
        self.adapt_pool = nn.AdaptiveAvgPool1d(50)
        self.fc = nn.Linear(32 * 50, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.adapt_pool(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_crop(tensor, fixed_length):
    # tensor shape: (n_mels, frames)
    if tensor.shape[-1] > fixed_length:
        return tensor[:, :fixed_length]
    elif tensor.shape[-1] < fixed_length:
        return F.pad(tensor, (0, fixed_length - tensor.shape[-1]))
    else:
        return tensor


def train_model(n_mels=80, groups=1, epochs=5, batch_size=32, device="cuda"):
    root = "./SpeechCommands"
    if not os.path.exists(root):
        os.makedirs(root)

    train_ds = YesNoDataset(root, subset="training")
    val_ds = YesNoDataset(root, subset="validation")
    test_ds = YesNoDataset(root, subset="testing")

    log_mel_transform = LogMelFilterBanks(n_mels=n_mels).to(device)

    def collate_fn(batch):
        waveforms, targets = zip(*batch)
        targets = torch.tensor(targets, dtype=torch.long)
        max_waveform_length = max([w.shape[-1] for w in waveforms])
        padded_waveforms = [F.pad(w, (0, max_waveform_length - w.shape[-1])) for w in waveforms]
        waveforms_padded = torch.stack(padded_waveforms).squeeze(1).to(device)
        log_mels = log_mel_transform(waveforms_padded)
        fixed_length = 100
        fixed_log_mels = [pad_crop(mel, fixed_length) for mel in log_mels]
        batched_log_mels = torch.stack(fixed_log_mels)
        return batched_log_mels, targets.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = SimpleCNN(n_mels=n_mels, groups=groups).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        total_loss /= len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                preds = model(x_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        val_acc = correct / total
        epoch_time = time.time() - start
        print(f"Epoch {epoch} | TrainLoss={total_loss:.4f} | ValAcc={val_acc:.4f} | Time={epoch_time:.2f}s")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    test_acc = correct / total
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Params: {count_params(model)}")

    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(model, (n_mels, 100), as_strings=True, print_per_layer_stat=False)
        print(f"FLOPs: {macs}")
    except ImportError:
        print("ptflops library not found")


if __name__ == "__main__":
    for nm in [20, 40, 80]:
        print(f"\nTraining with n_mels={nm}")
        train_model(n_mels=nm, groups=1, epochs=3)

    for g in [2, 4, 8, 16]:
        print(f"\nTraining with groups={g}")
        train_model(n_mels=80, groups=g, epochs=3)
