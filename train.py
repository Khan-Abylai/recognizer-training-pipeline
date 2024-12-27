import logging
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
import warnings
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import numpy as np
from torch.cuda import amp
from torch.nn.utils import clip_grad_norm_
from torch.nn import CTCLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.models import CRNN
from src.utils import load_weights, wer
from src.utils.converter import StrLabelConverter, RegionConverter
from src.dataset import LPDataset
import mlflow
import mlflow.pytorch
import os
from src.augmentation import transform_album, transform_old, transform_album_hard, no_transform
import string

warnings.filterwarnings("ignore")

logging.basicConfig(format='[%(asctime)s:] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', 
                    level=logging.INFO)

logger = logging.getLogger('training')


def train_one_epoch(rank, model, train_dataloader, criterion, ce_loss, optimizer, converter, epoch, num_epochs, region_converter=None):
    model.train()
    total_train_images = 0
    correct_train_predictions = 0
    correct_cls_predictions = 0
    word_error_rate = 0
    train_mean_loss = 0

    progress_bar = tqdm(train_dataloader) if rank == 0 else train_dataloader

    for idx, (images, labels, region, filepaths) in enumerate(progress_bar):
        if region_converter is not None:
            region_label = region_converter.encode(region).cuda()
        encoded_labels, length = converter.encode(labels)
        train_images = images.cuda()
        encoded_labels = encoded_labels.cuda()
        length = length.cuda()
        batch_size = length.shape[0]

        with amp.autocast():
            predictions, cls = model(train_images)
            predictions = predictions.permute(1, 0, 2).contiguous()
            prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)

            # Calculate loss
            current_loss = criterion(predictions, encoded_labels, prediction_size, length)
            current_loss_cls = ce_loss(cls, region_label)
        # Update metrics
        predictions = predictions.argmax(2).detach().cpu()
        predicted_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
        correct_train_predictions += (predicted_labels == labels).sum()
        if region_converter is not None:
            current_loss += current_loss_cls
            correct_cls_predictions += (cls.argmax(1) == region_label).sum()
        word_error_rate += wer(predicted_labels, labels)

        # Update running mean loss
        train_mean_loss = (
            train_mean_loss * (total_train_images / (total_train_images + batch_size)) +
            current_loss * batch_size / (total_train_images + batch_size)
        )

        total_train_images += batch_size

        # Log training metrics
        epoch_desc = (
            f'[{epoch}/{num_epochs}] Current Loss: {current_loss:.4f} '
            f'Loss: {train_mean_loss:.4f} Acc: {correct_train_predictions / total_train_images:.4f} Cls_Acc: {correct_cls_predictions/total_train_images:.4f} '
            f'WER: {word_error_rate / total_train_images:.4f}'
        )
        
        if rank == 0:
            progress_bar.set_description(epoch_desc)

        # Backpropagation
        model.zero_grad()
        clip_grad_norm_(model.parameters(), 1)
        current_loss.backward()
        optimizer.step()

    return train_mean_loss, correct_train_predictions, correct_cls_predictions, total_train_images, word_error_rate


def validate(model, val_dataloader, criterion, ce_loss, converter, region_converter=None):
    model.eval()
    total_val_images = 0
    correct_val_predictions = 0
    correct_val_classifier = 0
    val_mean_loss = 0

    with torch.no_grad():
        for val_images, val_labels, regions, val_filepaths in val_dataloader:
            if region_converter is not None:
                country_label_val = region_converter.encode(regions).cuda()
            encoded_val_labels, length = converter.encode(val_labels)
            predictions, cls = model(val_images.cuda())
            predictions = predictions.permute(1, 0, 2).contiguous()
            batch_size = length.shape[0]
            prediction_size = torch.LongTensor([predictions.size(0)]).repeat(batch_size)

            # Calculate validation loss
            current_val_loss = criterion(predictions, encoded_val_labels, prediction_size, length)

            predictions = predictions.argmax(2).detach().cpu()
            predicted_val_labels = np.array(converter.decode(predictions, prediction_size, raw=False))
            correct_val_predictions += (predicted_val_labels == val_labels).sum()
            if region_converter is not None:
                current_val_loss += ce_loss(cls, country_label_val)
                correct_val_classifier += (country_label_val == cls.argmax(1)).sum()
            # Update running mean loss
            val_mean_loss = (
                val_mean_loss * (total_val_images / (total_val_images + batch_size)) +
                current_val_loss * batch_size / (total_val_images + batch_size)
            )

            total_val_images += batch_size

    val_accuracy = correct_val_predictions / total_val_images
    return val_mean_loss, val_accuracy, correct_val_classifier / total_val_images


def train_model(rank, model, start_epochs, train_dataloader, val_dataloader, criterion, ce_loss, optimizer, converter, region_converter, num_epochs, model_directory, scheduler, ex_name):
    for epoch in range(start_epochs, num_epochs):
        train_mean_loss, correct_train_predictions, correct_cls_preds, total_train_images, word_error_rate = train_one_epoch(
            rank, model, train_dataloader, criterion, ce_loss, optimizer, converter, epoch, num_epochs, region_converter
        )
        
        val_mean_loss, val_accuracy, class_accuracy = validate(model, val_dataloader, criterion, ce_loss, converter, region_converter)

        train_accuracy = correct_train_predictions / total_train_images
        train_cls_accuracy = correct_cls_preds / total_train_images
        if rank == 0:
            epoch_description = (
                f'{epoch}/{num_epochs} Train: {train_mean_loss:.4f}, Accuracy: {train_accuracy:.4f}, Cls: {train_cls_accuracy:.4f}'
                f'Val: {val_mean_loss:.4f}, Accuracy: {val_accuracy:.4f}, Cls: {class_accuracy:.4f}, lr: {optimizer.param_groups[0]["lr"]}'
            )

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            model_path = os.path.join(model_directory, epoch_description.replace(" ", "_").replace("/", "_")+'.pth')
            torch.save(state, model_path)
            tqdm.write(epoch_description)

            mlflow.log_metric("train_loss", train_mean_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("train_class_accuracy", train_cls_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_mean_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            mlflow.log_metric("val_class_accuracy", class_accuracy, step=epoch)
            mlflow.log_metric("word_error_rate", word_error_rate, step=epoch)
            mlflow.pytorch.log_model(model, ex_name)

        scheduler.step(val_mean_loss)


def main(rank, config):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    global_cfg = config['global']
    alphabet = eval(global_cfg['alphabet'])
    distributed = global_cfg['distributed']
    
    if rank == 0:
        mlflow.set_tracking_uri("http://10.66.100.20:5000")
        mlflow.set_experiment(global_cfg["experiment_name"])
        mlflow.start_run(run_name=global_cfg["run_name"], run_id=global_cfg["run_id"])
        try:
            mlflow.log_params(config)
        except:
            print('mlflow params already logged')

    if distributed:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=torch.cuda.device_count(), rank=rank)
        torch.cuda.set_device(rank)
    
    model_cfg = config['model']
    model = CRNN(**model_cfg)
    model = model.cuda()
    if distributed:
        model = DistributedDataParallel(model, device_ids=[rank])
    
    model, start_epoch = load_weights(model, global_cfg['checkpoint'])
    if rank == 0:
        mlflow.pytorch.log_model(model, global_cfg["run_name"])
    config['scheduler']['min_lr'] = eval(config['scheduler']['min_lr'])
    optimizer = AdamW(model.parameters(), **config['optimizer'])
    scheduler = ReduceLROnPlateau(optimizer, **config['scheduler'], verbose=True)
    criterion = CTCLoss(reduction='sum', zero_infinity=True)
    ce_loss = torch.nn.CrossEntropyLoss()
    converter = StrLabelConverter(alphabet)

    config['train_data']['transform'] = eval(config['train_data']['transform'])
    config['val_data']['transform'] = eval(config['val_data']['transform'])
    region_converter = None
    if model_cfg['classification_regions'] is not None:
        region_converter = RegionConverter(model_cfg['classification_regions'])
    train_dataset = LPDataset(**config['train_data'], use_region=True if region_converter is not None else False, train=True)
    val_dataset = LPDataset(**config['val_data'], use_region=True if region_converter is not None else False)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=torch.cuda.device_count(),
                                                                    rank=rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=torch.cuda.device_count(),
                                                                  rank=rank, shuffle=False)
        train_dataloader = DataLoader(train_dataset, batch_size=config['loader']['batch_size'], shuffle=False,
                                  num_workers=config['loader']['num_workers'] // torch.cuda.device_count(), sampler=train_sampler,
                                  drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['loader']['batch_size'], shuffle=False,
                                    num_workers=config['loader']['num_workers'] // torch.cuda.device_count(), sampler=val_sampler,
                                    drop_last=True)
    else:
        train_dataloader = DataLoader(train_dataset, shuffle=True, **config['loader'])
        val_dataloader = DataLoader(val_dataset, shuffle=False, **config['loader'])

    train_model(rank, model, start_epoch, train_dataloader, val_dataloader,
                criterion, ce_loss, optimizer, converter, region_converter, global_cfg['epochs'],
                global_cfg['checkpoint'], scheduler, global_cfg['run_name'])

    # End the MLflow run
    mlflow.end_run()


if __name__ == '__main__':
    import yaml

    with open('train_config.yml', 'r') as f:
        cfg = yaml.safe_load(f)
    gpu = cfg['global']['gpu']
    distributed = cfg['global']['distributed']

    if not distributed:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        rank = 0
        main(rank, cfg)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        n_gpu = torch.cuda.device_count()
        mp.spawn(main, nprocs=n_gpu, args=(cfg,))