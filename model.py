import torch

import torch.nn as nn

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader, Dataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AutoEncoder(nn.Module):
    def __init__(self, input_dims):
        super(AutoEncoder, self).__init__()
        self.input_dims = input_dims
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dims),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z


class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
        self.fin_data = []
        self.make_tensor()

    def make_tensor(self):
        self.fin_data = torch.FloatTensor(self.data)

    def __len__(self):
        return len(self.fin_data)

    def __getitem__(self, idx):
        return self.fin_data[idx]


def train(train_x, epoch=16, batch_size=256):
    feature_num = len(train_x[0])
    train_dataset = MyDataset(train_x)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(feature_num).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    criterion_mse = nn.MSELoss().to(device)

    pbar = tqdm(range(epoch), desc="training")
    train_loss_list = []
    chunk = len(train_x) // 5
    for e in pbar:
        model.train()
        train_running_loss = 0.0
        for idx, datas in enumerate(train_loader):
            datas = datas.to(device)
            output, z = model(datas)
            loss = criterion_mse(output, datas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            if idx % chunk == 0:
                train_loss_list.append(train_running_loss / (idx + 1))
            pbar.set_postfix(epoch=f"{e + 1} | of {epoch}", loss=f"{loss:.5f}")
        print('Epoch: %d | Loss: %.4f' \
              % (e + 1, train_running_loss / (idx + 1)))

    return model


def test(test_x, model, batch_size=256):

    test_dataset = MyDataset(test_x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    latent_list = []
    rce_list = []
    for idx, datas in enumerate(tqdm(test_loader)):
        datas = datas.to(device)
        output, z = model(datas)
        latent_list += z.cpu().tolist()

        diff = output - datas
        sp_error_map = torch.sum(diff ** 2, dim=1) ** 0.5
        s = sp_error_map.size()
        sp_error_vec = sp_error_map.view(s[0], -1)
        recon_error = torch.mean(sp_error_vec, dim=-1)

        rce_list += recon_error.tolist()
    return rce_list
