import torch
import torch.nn as nn
from utils.train import Evaluate
from datasets.Data import Dataset_Mean_Std


class Client:
    """
    GeFL client.
    Reference: Kang et al., IEEE TMC 2025. Code: github.com/honggkang/hetero-model-fl-gen

    GAN: cGAN (conditional GAN), label conditioning via fill-map (D) and one-hot (G).
    Loss: pure BCE only (no auxiliary classifier loss).
    Optimizer states persisted per client across rounds (matching official optgs/optds).

    Each round:
      1. train_gan()    — train cGAN locally; return G/D states + optimizer states.
      2. train_target() — train target model on synthetic (local_ep_gen) then real (local_ep).
    """

    def __init__(self, model, local_train_loader, local_test_loader,
                 client_id, args, device='cpu'):
        self.client_id = client_id
        self.local_model = model
        self.train_loader = local_train_loader
        self.test_loader = local_test_loader
        self.device = device
        self.train_samples = len(local_train_loader.dataset)
        self.args = args
        self.n_iters = max(1, self.train_samples // args.batch_size)

        self.criterion = nn.CrossEntropyLoss()

        # Normalization params to convert GAN Tanh output [-1,1] → dataset space
        mean, std = Dataset_Mean_Std(args.dataset)
        self._gan_mean = torch.tensor(mean).view(1, -1, 1, 1)
        self._gan_std  = torch.tensor(std).view(1, -1, 1, 1)

        print(f"[GeFL] Client {client_id} | model={model.__class__.__name__} "
              f"| train={self.train_samples} | iters/epoch={self.n_iters}")

    def get_id(self):
        return self.client_id

    def _normalize_gen(self, imgs):
        """Convert GAN Tanh [-1,1] to the same space as the real dataset images."""
        mean = self._gan_mean.to(imgs.device)
        std  = self._gan_std.to(imgs.device)
        return ((imgs + 1) / 2 - mean) / std

    # ── GAN training ─────────────────────────────────────────────────────────

    def train_gan(self, generator, discriminator, gan_epochs, optg_state=None, optd_state=None):
        """
        Train cGAN locally for gan_epochs epochs.
        Labels fed to G as one-hot (B,10,1,1), to D as fill map (B,10,H,W).
        Loss: pure BCE only (no auxiliary CE).
        Optimizer states loaded/saved for continuity across rounds.
        Returns (G_state, D_state, optg_state, optd_state) all on CPU.
        """
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        generator.train()
        discriminator.train()

        g_opt = torch.optim.Adam(generator.parameters(), lr=self.args.gan_lr, betas=(self.args.b1, self.args.b2))
        d_opt = torch.optim.Adam(discriminator.parameters(), lr=self.args.gan_lr, betas=(self.args.b1, self.args.b2))
        if optg_state is not None:
            g_opt.load_state_dict(optg_state)
        if optd_state is not None:
            d_opt.load_state_dict(optd_state)

        num_classes = self.args.num_classes
        img_size = self.args.image_size if isinstance(self.args.image_size, int) else self.args.image_size[0]
        bce = nn.BCELoss().to(self.device)

        onehot = torch.zeros(num_classes, num_classes)
        onehot.scatter_(1, torch.arange(num_classes).view(-1, 1), 1)
        onehot = onehot.view(num_classes, num_classes, 1, 1)

        fill = torch.zeros(num_classes, num_classes, img_size, img_size)
        for i in range(num_classes):
            fill[i, i, :, :] = 1

        total_g_loss, total_d_loss, n_batches = 0.0, 0.0, 0

        for _ in range(gan_epochs):
            for real_imgs, labels in self.train_loader:
                mini_batch = real_imgs.size(0)
                real_imgs = real_imgs.to(self.device)

                y_real = torch.ones(mini_batch).to(self.device)
                y_fake = torch.zeros(mini_batch).to(self.device)
                y_fill_real = fill[labels].to(self.device)

                # ── Train D ──────────────────────────────────────────────────
                discriminator.zero_grad()
                d_real_loss = bce(discriminator(real_imgs, y_fill_real).view(-1), y_real)

                z = torch.randn(mini_batch, 100, 1, 1).to(self.device)
                fake_labels = (torch.rand(mini_batch, 1) * num_classes).long().view(-1)
                y_label_fake = onehot[fake_labels].to(self.device)
                y_fill_fake = fill[fake_labels].to(self.device)
                fake_imgs = generator(z, y_label_fake).detach()
                d_fake_loss = bce(discriminator(fake_imgs, y_fill_fake).view(-1), y_fake)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_opt.step()

                # ── Train G ──────────────────────────────────────────────────
                generator.zero_grad()
                z = torch.randn(mini_batch, 100, 1, 1).to(self.device)
                gen_labels = (torch.rand(mini_batch, 1) * num_classes).long().view(-1)
                y_label_gen = onehot[gen_labels].to(self.device)
                y_fill_gen = fill[gen_labels].to(self.device)
                gen_imgs = generator(z, y_label_gen)
                g_loss = bce(discriminator(gen_imgs, y_fill_gen).view(-1), y_real)
                g_loss.backward()
                g_opt.step()

                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                n_batches += 1

        g_state = {k: v.cpu() for k, v in generator.state_dict().items()}
        d_state = {k: v.cpu() for k, v in discriminator.state_dict().items()}
        generator.to('cpu')
        discriminator.to('cpu')
        torch.cuda.empty_cache()
        avg_g = total_g_loss / n_batches
        avg_d = total_d_loss / n_batches
        return g_state, d_state, g_opt.state_dict(), d_opt.state_dict(), avg_g, avg_d

    # ── Target model training ─────────────────────────────────────────────────

    def train_target(self, global_generator):
        """
        Sequential training matching official localUpdateTarget.py:
          Phase 1 (local_ep_gen epochs x n_iters): synthetic from global_generator.sample_image()
          Phase 2 (local_ep epochs):               real local data
        Fresh SGD optimizer per phase (matches official).
        GAN images are re-normalized from Tanh [-1,1] to dataset space before training.
        """
        self.local_model = self.local_model.to(self.device)
        self.local_model.train()

        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.args.gefl_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        # Phase 1: synthetic data
        if self.args.local_ep_gen > 0:
            global_generator = global_generator.to(self.device)
            global_generator.eval()

            for _ in range(self.args.local_ep_gen):
                for _ in range(self.n_iters):
                    with torch.no_grad():
                        imgs, labels = global_generator.sample_image(
                            self.args, sample_num=self.args.batch_size)
                    imgs = self._normalize_gen(imgs)
                    self.local_model.zero_grad()
                    logits = self.local_model(imgs)['logits']
                    loss = self.criterion(logits, labels.argmax(dim=1))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10.0)
                    optimizer.step()

            global_generator.to('cpu')
            torch.cuda.empty_cache()

        # Phase 2: real data — fresh optimizer (matches official, resets momentum)
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.args.gefl_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        for _ in range(self.args.local_ep):
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                logits = self.local_model(imgs)['logits']
                loss = self.criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10.0)
                optimizer.step()

        self.local_model.to('cpu')
        self.local_model.eval()
        torch.cuda.empty_cache()

    # ── Model upload/download for grouped FedAvg ──────────────────────────────

    def get_model_state(self):
        return self.local_model.state_dict(), self.train_samples

    def set_model_state(self, state_dict):
        self.local_model.load_state_dict(state_dict)

    def evaluate(self):
        loss, acc = Evaluate(self.local_model, self.test_loader, device=self.device)
        return loss, acc
