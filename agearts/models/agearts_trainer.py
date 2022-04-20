import os.path
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


class AgeArtsTrainer:
    def __init__(self, model, criterion, optimizer,
                 n_epochs, trainloader, validloader, testloader,
                 root_to_save_model, config, scheduler,
                 device=torch.device('cuda'), verbose=True, save_info_txt=False,
                 earlystopping=True, es_delta=0, es_patience=5, es_mode='loss'
                 ):
        self.config = config
        self.device = device
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.verbose = verbose
        self.save_info_txt = save_info_txt
        self.earlystopping = earlystopping
        self.es_patience = es_patience
        self.es_delta = es_delta
        self.epoch = 1
        self.early_stop = False
        self.es_counter = 0
        self.es_val_loss_min = np.Inf
        self.root_to_save_model = root_to_save_model
        self.es_mode = es_mode

        # ---- for mixed precision
        self.scaler = GradScaler()
        # ----

    def _step(self, data):
        images, labels = data['img'], data['age']

        labels = labels.to(torch.float32)

        inputs, labels = images.to(self.device), labels.to(self.device)

        with autocast():
            outputs = self.model(inputs).reshape(-1,)

            # print('OUTPUTS: ', outputs * 2012 + 1401)
            # print('LABELS: ', labels * 2012 + 1401)

            loss = self.criterion(outputs.reshape(-1, 1), labels)

        return loss

    def _train_step(self):
        self.model.train()

        losses = []

        for i, data in enumerate(tqdm(self.trainloader)):
            loss = self._step(data)

            self.optimizer.zero_grad()

            # loss.backward()
            #
            # self.optimizer.step()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            losses.append(loss.item())

            if i % 100 == 0:
                text = f'{self.epoch} epoch; {i} batch) train loss: {np.average(losses)}'
                self._save_info_about_training(text, file_name='losses.txt')

            self.scaler.update()
        return np.average(losses)

    def _valid_step(self):
        self.model.eval()

        losses = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(self.validloader)):
                loss = self._step(data)
                losses.append(loss.item())
                if i % 100 == 0:
                    text = f'{self.epoch} epoch; {i} batch) valid loss: {np.average(losses)}'
                    self._save_info_about_training(text, file_name='losses.txt')
        return np.average(losses)

    def train_model(self):
        for self.epoch in tqdm(range(self.epoch, self.n_epochs + 1)):

            train_loss = self._train_step()
            valid_loss = self._valid_step()

            self._print_step_statistic(train_loss, valid_loss)

            if self.scheduler:
                self.scheduler.step(valid_loss)

            if self.es_mode == 'loss':
                self._early_stopping_valid_loss(valid_loss)

            if self.early_stop:
                if self.verbose:
                    print_msg = f'Early stopping at {self.epoch} epoch'
                    print(print_msg)
                    self._save_info_about_training(text=print_msg)
                break

        self.load_model()

    def _print_step_statistic(self, train_loss, valid_loss):
        if self.verbose:
            epoch_len = len(str(self.n_epochs))
            print_msg = (f'[{self.epoch:>{epoch_len}}/{self.n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f} '
                         )

            self._save_info_about_training(text=print_msg)
            print(print_msg)

    def _early_stopping_valid_loss(self, indicator):
        if self.earlystopping:
            if type(indicator) == np.array:
                if self.es_val_loss_min is np.Inf:
                    self.es_val_loss_min = indicator
                    self.save_checkpoint()
                elif sum(indicator < self.es_val_loss_min - self.es_delta) / len(indicator) > 0.5:
                    self.es_counter += 1
                    if self.verbose:
                        print_msg = f'EarlyStopping counter: {self.es_counter} out of {self.es_patience}'
                        print(print_msg)
                        self._save_info_about_training(text=print_msg)
                    if self.es_counter >= self.es_patience:
                        self.early_stop = True
                else:
                    self.es_val_loss_min = indicator
                    self.save_checkpoint()
                    self.es_counter = 0
                    self.val_loss_min = indicator
            else:
                if self.es_val_loss_min is np.Inf:
                    self.es_val_loss_min = indicator
                    self.save_checkpoint()
                elif indicator > self.es_val_loss_min - self.es_delta:
                    self.es_counter += 1
                    if self.verbose:
                        print_msg = f'EarlyStopping counter: {self.es_counter} out of {self.es_patience}'
                        print(print_msg)
                        self._save_info_about_training(text=print_msg)
                    if self.es_counter >= self.es_patience:
                        self.early_stop = True
                else:
                    self.es_val_loss_min = indicator
                    self.save_checkpoint()
                    self.es_counter = 0
                    self.val_loss_min = indicator

    def _save_info_about_training(self, text, file_name='info_about_training.txt'):
        if self.save_info_txt:
            if not os.path.exists(self.root_to_save_model):
                os.makedirs(self.root_to_save_model)
            full_path = os.path.join(self.root_to_save_model, file_name)
            with open(full_path, "a") as file:
                file.write(text + '\n')

    def save_checkpoint(self, file_name='checkpoint.pt'):
        if self.verbose:
            print('Saving model')
        if not os.path.exists(self.root_to_save_model):
            os.makedirs(self.root_to_save_model)
        full_path_file = os.path.join(self.root_to_save_model, file_name)

        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            # 'grad_scaler': self.scaler.state_dict(),
            'config': self.config,
        }, full_path_file)

    def load_model(self, file_name='checkpoint.pt'):
        full_path_file = os.path.join(self.root_to_save_model, file_name)

        checkpoint = torch.load(full_path_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.scaler.load_state_dict(checkpoint['grad_scaler'])
        self.epoch = checkpoint['epoch']
