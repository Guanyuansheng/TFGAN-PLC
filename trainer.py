import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import myDataset
from models.tfgan.generator import Generator
from models.tfgan.discriminator import Discriminator
from models.stft_loss import MultiResolutionSTFTLoss
from optimizsers import RAdam


def get_pesq(ref, deg):
    return pesq(16000, ref, deg, 'nb')

def pingjie(inputs, frame_size, frame_shift):
    nframes = inputs.shape[0]
    sig_len = (nframes - 1) * frame_shift + frame_size
    sig = torch.zeros([sig_len,]).cuda()
    ones = torch.zeros_like(sig).cuda()
    start = 0
    end = start + frame_size
    for i in range(nframes):
        sig[start:end] += inputs[i, :].squeeze(0)
        ones[start:end] += 1
        start = start + frame_shift
        end = start + frame_size
    return sig / ones


class TFGAN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        if not os.path.exists(self.hparams.output_path):
            os.makedirs(self.hparams.output_path)

        self.generator = Generator()
        self.discriminator = Discriminator()
        # self.stft_mag = MultiSTFTMag()

    def forward(self, noisy):
        """
        Generates a speech using the generator
        given input noise z and noisy speech
        """
        return self.generator(noisy)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for RaLSGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(self.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        noisy, clean, lens = batch

        # generate speechs
        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
        g_out = self.generator(noisy)
        # g_out_mags, clean_mags = self.stft_mag(g_out.squeeze(1), clean.squeeze(1))

        # make discriminator
        disc_real = self.discriminator(clean)
        disc_fake = self.discriminator(g_out)

        # Summarize
        tensorboard = self.logger.experiment
        if batch_idx % 200 == 0:
            tensorboard.add_audio('inpainted speech', pingjie(g_out, self.hparams.slice_len, self.hparams.shift), batch_idx, 16000)
            tensorboard.add_audio('uninpaint speech', pingjie(noisy, self.hparams.slice_len, self.hparams.shift), batch_idx, 16000)
            tensorboard.add_audio('clean speech', pingjie(clean, self.hparams.slice_len, self.hparams.shift), batch_idx, 16000)
            pesq = get_pesq(pingjie(clean, self.hparams.slice_len, self.hparams.shift).cpu().detach().numpy(),
                            pingjie(g_out, self.hparams.slice_len, self.hparams.shift).cpu().detach().numpy())
            tensorboard.add_scalar('pesq', pesq, batch_idx)
            print(batch_idx, 'batch_idx pesq=', pesq)

        # RaLSGAN-GP
        real_logit = disc_real - torch.mean(disc_fake)
        fake_logit = disc_fake - torch.mean(disc_real)

        # Train generator
        if optimizer_idx % 2 == 0:
            g_loss = (torch.mean((real_logit + 1.) ** 2) + torch.mean((fake_logit - 1.) ** 2)) / 2
            # l1_loss
            l1_loss = 100 * F.mse_loss(g_out, clean, reduction='mean')
            # stft_loss
            stft_loss = MultiResolutionSTFTLoss()
            g_stft = stft_loss(torch.squeeze(g_out, 1), torch.squeeze(clean, 1))
            g_loss = g_loss + l1_loss + 0.5 * g_stft
            g_loss.requires_grad_(True)

            tensorboard.add_scalar('g_loss', g_loss, batch_idx)
            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
            return g_loss

        # train discriminator
        d_loss = (torch.mean((real_logit - 1.) ** 2) + torch.mean((fake_logit + 1.) ** 2))/2
        d_loss.requires_grad_(True)
        # gradient_penalty = self.compute_gradient_penalty(clean, g_out)
        # energy loss
        # d_loss = d_loss + gradient_penalty

        tensorboard.add_scalar('d_loss', d_loss, batch_idx)
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        return d_loss

    def validation_step(self, batch, batch_idx):
        noisy, clean, lens = batch

        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
        generated = self.forward(noisy)
        l1_loss = 100 * torch.mean(torch.abs(clean - generated))
        pesq = get_pesq(pingjie(clean, self.hparams.slice_len, self.hparams.shift).cpu().detach().numpy(),
                        pingjie(generated, self.hparams.slice_len, self.hparams.shift).cpu().detach().numpy())

        output = {
            'loss': l1_loss,
            'pesq': pesq
        }
        return output

    def test_step(self, batch, batch_idx):
        noisy, clean, lens = batch

        # z = nn.init.uniform_(torch.Tensor(noisy.shape[0], 100), -1., 1.)
        # z = z.type_as(noisy)
        test = self.forward(noisy)

        save_path_clean = os.path.join(self.hparams.output_path, 'clean/test_audio_%d.wav' % batch_idx)
        save_path_test = os.path.join(self.hparams.output_path, 'inpainted/test_audio_%d.wav' % batch_idx)
        save_path_noisy = os.path.join(self.hparams.output_path, 'noisy/test_audio_%d.wav' % batch_idx)
        print(save_path_test)
        torchaudio.save(save_path_clean, pingjie(clean, self.hparams.slice_len, self.hparams.shift).cpu(), 16000)
        torchaudio.save(save_path_test, pingjie(test, self.hparams.slice_len, self.hparams.shift).cpu(), 16000)
        torchaudio.save(save_path_noisy, pingjie(noisy, self.hparams.slice_len, self.hparams.shift).cpu(), 16000)
        # torchaudio.save(save_path_test, test, 16000)
        print('Successfully inpainted %d audios' % (batch_idx+1))

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer_g = RAdam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
        optimizer_d = RAdam(self.discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))
        # optimizer_stftd = RAdam(self.dc_discriminator.parameters(), lr=0.00005, betas=(0.5, 0.9))

        return [optimizer_g, optimizer_d], []

    def collate_fn(self, batch):
        noisy = torch.cat([item[0].unsqueeze(1) for item in batch], 0)
        # print('noisy size = ', noisy.size())
        clean = torch.cat([item[1].unsqueeze(1) for item in batch], 0)
        lens = torch.LongTensor([item[2] for item in batch])
        return noisy, clean, lens

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            myDataset(self.hparams.data_dir, self.hparams.slice_len, loss_rate='loss_noisy'),
            batch_size=self.hparams.batch_size, collate_fn=self.collate_fn, shuffle=False, num_workers=4)

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            myDataset(self.hparams.val_data_dir, self.hparams.slice_len, loss_rate='loss_10'),
            batch_size=1, collate_fn=self.collate_fn, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(
            myDataset(self.hparams.test_data_dir, self.hparams.slice_len, loss_rate='loss_10'),
            batch_size=1, collate_fn=self.collate_fn, shuffle=False, num_workers=4)


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--shift', default=640, type=int)
    parser.add_argument('--slice_len', default=2560, type=int)
    # path
    parser.add_argument('--data_dir', default='/media/guanyuansheng/dataset/clean_trainset_56spk_wav_16k', type=str)
    parser.add_argument('--val_data_dir', default='/media/guanyuansheng/dataset/clean_trainset_56spk_wav_16k/validation', type=str)
    parser.add_argument('--output_path', default='./outputs', type=str)
    parser.add_argument('--test_data_dir', default='/media/guanyuansheng/dataset/clean_trainset_56spk_wav_16k/test', type=str)
    # parse params
    hparams = parser.parse_args()

    model = TFGAN(hparams)

    checkpoint_callback = ModelCheckpoint(filepath='./checkpoints', save_top_k=3, verbose=True,
                                          monitor='pesq', mode='max', prefix='pesq')

    tb_logger = pl_loggers.TensorBoardLogger('./logs/')
    print('Training has started. Please use \'tensorboard --logdir=./logs\' to monitor.')

    trainer = pl.Trainer(gpus='3', checkpoint_callback=checkpoint_callback,
                        resume_from_checkpoint='./checkpoints/pesq-epoch=103.ckpt',
                        progress_bar_refresh_rate=10,
                        logger=tb_logger, profiler=True)

    if hparams.mode == 'train':
        trainer.fit(model)
    elif hparams.mode == 'test':
        model.freeze()
        trainer.test(model)
