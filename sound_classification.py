import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchaudio import transforms
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

PATH_TO_DATASET = 'UrbanSound8K/audio/'
PATH_TO_ANNOTATION = 'UrbanSound8K/metadata/UrbanSound8K.csv'
SAMPLE_RATE = 22050
NUM_OF_SAMPLES = 22050
SHUFFLE = True
device = torch.device('cuda')


mel_spec = transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
spec = transforms.Spectrogram(n_fft=512, hop_length=256)
mfcc = transforms.MFCC(sample_rate=SAMPLE_RATE, log_mels=True, n_mfcc=128)


class UrbanSound(Dataset):
    def __init__(self, PATH_TO_DATASET, PATH_TO_ANNOTATION, sample_rate, num_samples, transform, device):
        self.annotations = pd.read_csv(PATH_TO_ANNOTATION)
        self.audio_dir = PATH_TO_DATASET
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device
        self.transform = transform.to(self.device)

    def __getitem__(self, idx):
        audio_path = self.audio_dir + 'fold' + str(self.annotations.iloc[idx, 5]) + '/' + \
                     self.annotations.iloc[idx, 0]
        label = self.annotations.iloc[idx, 6]
        signal, sr = torchaudio.load(audio_path, normalize=True)
        signal = signal.to(self.device)
        signal = self._resample_(signal, sr).to(self.device)
        signal = self._to_mono_(signal)
        signal = self._pad_or_cut_(signal, self.num_samples)
        # plt.figure()
        # plt.plot(signal[0].cpu())
        # plt.show()

        # signal = self._normalize(signal)
        signal = self.transform(signal)
        # signal = torchaudio.transforms.AmplitudeToDB()(signal)
        # plt.figure()
        # plt.specgram(signal[0].cpu(), Fs=self.sample_rate)
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')
        # plt.show()

        # plt.figure()
        # plt.imshow(signal[0].cpu())
        # plt.show()

        return signal, label

    def __len__(self):
        return len(self.annotations)

    def _normalize(self, signal):
        signal = (signal[0] - torch.min(signal[0])) / (torch.max(signal[0]) - torch.min(signal[0]))
        signal = signal.unsqueeze(dim=0)
        # signal = torch.log10(torch.abs(signal[0]) + 1e-10)
        # signal = signal.unsqueeze(dim=0)
        return signal

    def _resample_(self, signal, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _pad_or_cut_(self, signal, num_samples):
        if signal.shape[1] < num_samples:
            # plt.figure()
            # plt.imshow(torchaudio.transforms.AmplitudeToDB()(self.transform(signal)[0].cpu()))
            # plt.show()
            length = signal.shape[1]
            needed_samples = num_samples - length

            '''
                padding with reflection --> Bad result
            '''
            # if needed_samples >= length:
            #     signal = torch.nn.functional.pad(signal, (0, length-1), mode='reflect')
            #     signal = self._pad_or_cut_(signal, num_samples)
            #     return signal
            # signal = torch.nn.functional.pad(signal, (0, needed_samples), mode='reflect')

            '''
                Normal padding (zero extend)
            '''
            signal = torch.nn.functional.pad(signal, (0, needed_samples))

            # plt.figure()
            # plt.imshow(torchaudio.transforms.AmplitudeToDB()(self.transform(signal)[0].cpu()))
            # plt.show()
        elif signal.shape[1] > num_samples:
            # plt.figure()
            # plt.imshow(torchaudio.transforms.AmplitudeToDB()(self.transform(signal)[0].cpu()))
            # plt.show()
            signal = signal[:, :num_samples]
            # plt.figure()
            # plt.imshow(torchaudio.transforms.AmplitudeToDB()(self.transform(signal)[0].cpu()))
            # plt.show()
        return signal

    def _to_mono_(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear = nn.Linear(in_features=1152, out_features=10)
        self.do = nn.Dropout(0.4)
        # self.linear2 = nn.Linear(in_features=1024, out_features=64)
        # self.linear3 = nn.Linear(in_features=64, out_features=10)

        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        # x = self.do(x)
        x = self.conv2(x)
        # x = self.do(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # x = self.conv8(x)
        # x = self.conv9(x)
        x = self.flatten(x)
        y = self.linear(x)
        # y = self.linear2(y1)
        # y = self.linear3(y1)

        return y


usd = UrbanSound(PATH_TO_DATASET, PATH_TO_ANNOTATION, SAMPLE_RATE, NUM_OF_SAMPLES, mfcc, device)
train_set, test_set = torch.utils.data.random_split(usd, [0.75, 0.25], generator=torch.Generator().manual_seed(2147483647))
if __name__ == '__main__':
    # print(torch.cuda.is_available())
    #
    # print(len(train_set), len(test_set))
    # train_loader = DataLoader(train_set, batch_size=128, shuffle=SHUFFLE)

    # model = CNN().to(device)
    # n_epochs = 25
    # criterion = nn.CrossEntropyLoss().to(device)
    # criterion2 = nn.L1Loss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    trainingEpoch_loss = []
    validationEpoch_loss = []

    # for fold in range(1, 11):
    #     print('FOLD', fold)
    train_set_idx = []
    test_set_idx = []
    model = CNN().to(device)
    n_epochs = 10
    criterion = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1)

    # for i in range(len(usd)):
    #     if usd.annotations.iloc[i, 5] == fold:
    #         test_set_idx.append(i)
    #     else:
    #         train_set_idx.append(i)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=SHUFFLE)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=SHUFFLE)

    for epoch in range(n_epochs):
        arr = []
        model.train()
        print('epoch:', epoch)
        batch_ctr = 1
        for input, target in train_loader:
            batch_ctr += 1
            input = input.to(device)
            target = target.long()
            target = target.to(device)
            predicted = model(input)
            # print(predicted2)
            # print(criterion(predicted1, target), criterion(predicted2, target))
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optimizer.step()
            arr.append(loss.item())
            print('loss:', loss.item())
        print('epoch:', epoch, 'loss:', sum(arr)/len(arr))
        trainingEpoch_loss.append(np.array(arr).mean())
        # if sum(arr)/len(arr) < 0.1:
        #     torch.save(model.state_dict(), 'ckpt_' + str(epoch) + '.pth')

        model.eval()
        for input, target in test_loader:
            validationStep_loss = []
            input = input.to(device)
            target = target.long()
            target = target.to(device)
            pred = model(input)
            validation_loss = criterion(pred, target)
            validationStep_loss.append(validation_loss.item())
        validationEpoch_loss.append(np.array(validationStep_loss).mean())
        print('epoch:', epoch, 'val loss:', np.array(validationStep_loss).mean())

    torch.save(model.state_dict(), 'ckpt9.pth')

    from matplotlib import pyplot as plt

    plt.plot(trainingEpoch_loss, label='train_loss')
    plt.plot(validationEpoch_loss, label='val_loss')
    plt.legend()
    plt.show()



