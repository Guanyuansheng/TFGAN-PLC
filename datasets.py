import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import os
import glob
import random

class myDataset(Dataset):
    def __init__(self, data_dir, slice_len, loss_rate='loss_20'):
        self.data_dir = data_dir
        self.loss_rate = loss_rate
        print('Training data set is', self.loss_rate)
        self.clean_root = os.path.join(self.data_dir, 'real')
        self.noisy_root = os.path.join(self.data_dir, self.loss_rate)

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)

        self.slice_len = slice_len

    def get_path(self, root):
        paths = glob.glob(os.path.join(root, '*.wav'))
        return sorted(paths)

    # def padding(self, x):
    #     x.unsqueeze_(0)
    #     len_x = x.size(-1)
    #     pad_len = self.stride - len_x % self.stride
    #     return F.pad(x, (pad_len, 0), mode='constant')

    def truncate(self, n, c):
        offset = 2560
        length = n.size(-1)
        start = torch.randint(length - offset, (1,))
        return n[:, start:start + offset], c[:, start:start + offset]

    def signal_to_frame(self, c, n, frame_size=2560, frame_shift=2560):
        sig_len = len(c)
        nframes = (sig_len // frame_shift)
        c_slice = np.zeros([nframes, frame_size])
        n_slice = np.zeros([nframes, frame_size])
        start = 0
        end = start + frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                c_slice[i, :] = c[start:end]
                n_slice[i, :] = n[start:end]
                k += 1
            else:
                tail_size = sig_len - start
                c_slice[i, :tail_size] = c[start:]
                n_slice[i, :tail_size] = n[start:]
            start = start + frame_shift
            end = start + frame_size
        return c_slice, n_slice

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        clean = sf.read(self.clean_path[idx])[0]
        noisy = sf.read(self.noisy_path[idx])[0]

        clean_slice, noisy_slice = self.signal_to_frame(clean, noisy, self.slice_len, 2560)
        length = clean_slice.shape[1]
        clean = torch.FloatTensor(clean_slice)
        noisy = torch.FloatTensor(noisy_slice)

        # clean = self.padding(clean).squeeze(0)
        # noisy = self.padding(noisy).squeeze(0)

        # clean /= clean.abs().max()
        # noisy /= noisy.abs().max()

        return noisy, clean, length
