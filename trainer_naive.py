import os
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
import numpy as np
import shutil
import copy
import csv
import queue
import time

from torch.nn import functional as F

from data_utils.transformer_2d import Get_ROI, RandomFlip2D, RandomRotate2D, RandomErase2D, RandomAdjust2D, RandomDistort2D, RandomZoom2D, RandomNoise2D
from data_utils.data_loader import DataGenerator, CropResize, To_Tensor, Trunc_and_Normalize
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


import random
import warnings
warnings.filterwarnings('ignore')
import setproctitle

from utils import dfs_remove_weight

# GPU version.


class SemanticSeg(object):
    '''
    Control the training, evaluation, and inference process.
    Args:
    - net_name: string
    - lr: float, learning rate.
    - n_epoch: integer, the epoch number
    - channels: integer, the channel number of the input
    - num_classes: integer, the number of class
    - input_shape: tuple of integer, input dim
    - crop: integer, cropping size
    - batch_size: integer
    - num_workers: integer, how many subprocesses to use for data loading.
    - device: string, use the specified device
    - pre_trained: True or False, default False
    - weight_path: weight path of pre-trained model
    '''
    def __init__(self,
                 net_name=None,
                 encoder_name=None,
                 sampler_name=None,
                 lr=1e-3,
                 n_epoch=100,
                 warmup_epoch=5,
                 sample_inteval=5,
                 channels=1,
                 num_classes=2,
                 target_names=None,
                 max_percent=0.5,
                 init_percent=0.1,
                 roi_number=1,
                 scale=None,
                 input_shape=None,
                 crop=48,
                 batch_size=6,
                 num_workers=4,
                 device=None,
                 pre_trained=False,
                 ex_pre_trained=False,
                 ckpt_point=True,
                 seg_weight_path=None,
                 weight_decay=0.,
                 momentum=0.95,
                 gamma=0.1,
                 milestones=[40, 80],
                 T_max=5,
                 mean=None,
                 std=None,
                 topk=50,
                 use_fp16=True):
        super(SemanticSeg, self).__init__()

        self.net_name = net_name
        self.encoder_name = encoder_name
        self.sampler_name = sampler_name
        self.lr = lr
        self.n_epoch = n_epoch
        self.warmup_epoch = warmup_epoch
        self.sample_inteval = sample_inteval
        self.channels = channels
        self.num_classes = num_classes
        self.target_names = target_names
        self.max_percent = max_percent
        self.init_percent = init_percent
        self.roi_number = roi_number
        self.scale = scale
        self.input_shape = input_shape
        self.crop = crop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.pre_trained = pre_trained
        self.ex_pre_trained = ex_pre_trained
        self.ckpt_point = ckpt_point
        self.seg_weight_path = seg_weight_path

        self.start_epoch = 0
        self.global_step = 0
        self.seg_loss_threshold = 1.0
        self.metrics_threshold = 0.

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.gamma = gamma
        self.milestones = milestones
        self.T_max = T_max
        self.mean = mean
        self.std = std
        self.topk = topk
        self.use_fp16 = use_fp16

        os.environ['CUDA_VISIBLE_DEVICES'] = self.device

        self.net = self._get_seg_net(self.net_name)

        if self.pre_trained:
            self._get_pre_trained_seg_net(self.seg_weight_path, ckpt_point)

        if self.roi_number is not None:
            assert self.num_classes == 2, "num_classes must be set to 2 for binary segmentation"

        self.get_roi = False

    def trainer(self,
                train_path,
                val_path,
                cur_fold,
                output_dir=None,
                semi_save_dir=None,
                semi_factor=1.0,
                semi_delta=5e-5,
                log_dir=None,
                optimizer='Adam',
                seg_loss_fun='Cross_Entropy',
                sample_mode='linear',
                sample_from_all_data=True,
                class_weight=None,
                lr_scheduler=None,
                freeze_encoder=False,
                get_roi=False,
                repeat_factor=1.0,
                sample_strategy='norm',
                sample_patience=10,
                sample_times=10,
                args=None):

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.cuda.manual_seed_all(0)
        print('Device:{}'.format(self.device))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        output_dir = os.path.join(output_dir, "fold" + str(cur_fold))
        log_dir = os.path.join(log_dir, "fold" + str(cur_fold))

        if os.path.exists(log_dir):
            if not self.pre_trained:
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
        else:
            os.makedirs(log_dir)

        if os.path.exists(output_dir):
            if not self.pre_trained:
                shutil.rmtree(output_dir)
                os.makedirs(output_dir)
        else:
            os.makedirs(output_dir)
        
        if self.sampler_name is not None and 'ceal' in self.sampler_name:
            semi_save_dir = os.path.join(semi_save_dir, "fold" + str(cur_fold))
            if os.path.exists(semi_save_dir):
                if not self.pre_trained:
                    shutil.rmtree(semi_save_dir)
                    os.makedirs(semi_save_dir)
            else:
                os.makedirs(semi_save_dir)

        def record_sample_list(record_csv,key,sample_list=[]):
            import pandas as pd
            if not os.path.exists(record_csv):
                df = pd.DataFrame(columns=[str(i) for i in range(self.n_epoch)])
                df['col'] = [np.nan] * len(train_path)
                df.to_csv(record_csv,index=False)
            if len(sample_list) != 0:
                df = pd.read_csv(record_csv)
                df[key][:len(sample_list)] = [os.path.basename(case) for case in sample_list]
                df.to_csv(record_csv,index=False)

        segnet_output_dir = os.path.join(output_dir, 'segnet')

        if not os.path.exists(segnet_output_dir):
            os.makedirs(segnet_output_dir)

        # self.step_per_epoch = len(train_path) // self.batch_size
        self.writer = SummaryWriter(log_dir)
        if not self.pre_trained:
            if args is not None and isinstance(args,dict):
                for key, value in args.items():
                    self.writer.add_text(key,str(value),0)

        net = self.net

        if freeze_encoder:
            for param in net.encoder.parameters():
                param.requires_grad = False

        lr = self.lr
        loss = self._get_seg_loss(seg_loss_fun, class_weight)

        if len(self.device.split(',')) > 1:
            net = DataParallel(net)

        # copy to gpu
        net = net.cuda()
        loss = loss.cuda()

        # optimizer setting
        optimizer = self._get_optimizer(optimizer, net, lr)
        scaler = GradScaler()

        if lr_scheduler is not None:
            lr_scheduler = self._get_lr_scheduler(lr_scheduler, optimizer)
            
        # loss_threshold = 1.0
        early_stopping = EarlyStopping(patience=40,
                                       verbose=True,
                                       delta=1e-3,
                                       monitor='val_run_dice',
                                       op_type='max')
        
        # dataloader setting
        data_size = len(train_path)
        self.repeat_factor = repeat_factor
        self.get_roi = get_roi

        if not self.pre_trained:
            with open(os.path.join(log_dir,'record.csv'), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                rows = ['epoch','label_ratio','unlabel_ratio','train_run_dice','val_run_dice']
                rows.extend(self.target_names*2)
                csvwriter.writerow(rows)

        if self.sampler_name is not None:
            self.sampler = self._get_sampler(self.sampler_name)
            if not self.pre_trained:
                self.unlabeled_data_pool = copy.deepcopy(train_path)
                self.labeled_data_pool = []
            # sample_times = int((self.n_epoch - self.warmup_epoch)/self.sample_inteval)
            self.sample_times = sample_times
            self.samples_per_epoch = self._get_samples_per_epoch(
                N_sample=len(train_path),
                sample_times=self.sample_times,
                sample_mode=sample_mode)

            print(self.samples_per_epoch)
            print('total sample num = %d'% sum(self.samples_per_epoch))
            self.sample_queue = queue.Queue()
            for item in self.samples_per_epoch[1:]:
                self.sample_queue.put(item)
        
        else:
            self.unlabeled_data_pool = []
            self.labeled_data_pool = copy.deepcopy(train_path)


        val_loader = self._get_data_loader(val_path,'val',repeat_factor=1.0)
        train_loader = self._get_data_loader(train_path,'train',repeat_factor=self.repeat_factor)
        total_sample_time = 0
        sample_count = 0
        for epoch in range(self.start_epoch, self.n_epoch):

            setproctitle.setproctitle('{}: {}/{}'.format('User', epoch, self.n_epoch))

            sample_flag = False
            labeled_data = []
            semi_data = []
            if self.sampler_name is not None:
                if epoch == 0:
                    labeled_data = self._random_sampling(sample_pool=train_path)
            
                else:
                    if sample_strategy == 'iq':
                        if early_stopping.counter >= sample_patience and not self.sample_queue.empty():
                            sample_flag = True
                            early_stopping.counter = 0
                    else:
                        if epoch >= self.warmup_epoch and ((epoch - self.warmup_epoch) % self.sample_inteval == 0) \
                            and not self.sample_queue.empty():
                            sample_flag = True

                if sample_flag:
                    if not sample_from_all_data:
                        unlabeled_data_pool = random.sample(self.unlabeled_data_pool, k=int(len(self.unlabeled_data_pool) * 0.1)) # 0.1 can be set to another value
                    else:
                        unlabeled_data_pool = self.unlabeled_data_pool
                    
                    if 'kcenter' in self.sampler_name:
                        sample_loader = self._get_data_loader(unlabeled_data_pool + self.labeled_data_pool,'val',repeat_factor=1.0)
                    else:
                        sample_loader = self._get_data_loader(unlabeled_data_pool,'val',repeat_factor=1.0)
                        
                    sample_nums = self.sample_queue.get()
                    if sample_nums != 0:
                        sample_count += 1
                        s_time = time.time()
                        print(f'************* start sampling : {sample_nums}, count:{sample_count}/{self.sample_times} *************')
                        if 'ceal' in self.sampler_name:
                            labeled_data, semi_data = self.sampler(
                                seg_net=net, 
                                unlabeled_data_pool=unlabeled_data_pool,
                                sample_loader=sample_loader,
                                sample_nums=sample_nums,
                                semi_save_dir=os.path.join(semi_save_dir,f'epoch_{epoch}'),
                                delta=semi_delta)
                            # update delta
                            semi_delta = semi_delta - 0.033 * 1e-5 * sample_count
                        else:
                            labeled_data = self.sampler(
                                seg_net=net, 
                                unlabeled_data_pool=unlabeled_data_pool,
                                sample_loader=sample_loader,
                                sample_nums=sample_nums)
                        
                        print(f'************* finish sampling : {sample_nums} *************')
                        total_sample_time += time.time() - s_time
                        print('sample time:%.4f' % (time.time() - s_time))

                if len(labeled_data) != 0:
                    self.labeled_data_pool.extend(labeled_data)    
                    self._update_unlabeled_data_pool(labeled_data)
                    random.shuffle(self.labeled_data_pool)
                    if len(semi_data) != 0:
                        random.shuffle(semi_data)
                        train_loader = self._get_data_loader(self.labeled_data_pool + semi_data,'train',repeat_factor=self.repeat_factor)
                    else:
                        train_loader = self._get_data_loader(self.labeled_data_pool,'train',repeat_factor=self.repeat_factor)

            train_loss, train_dice, train_run_dice = self._train_on_epoch(
                epoch=epoch,
                net=net,
                criterion=loss,
                optimizer=optimizer,
                scaler=scaler,
                train_loader=train_loader)
            
            val_loss, val_dice, val_run_dice = self._val_on_epoch(
                epoch=epoch,
                net=net,
                criterion=loss,
                val_loader=val_loader)

            if lr_scheduler is not None:
                lr_scheduler.step()
            

            torch.cuda.empty_cache()

            print('epoch:{},train_loss:{:.5f},val_loss:{:.5f}'.format(epoch, train_loss, val_loss))

            print('epoch:{},train_dice:{:.5f},train_run_dice:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
                .format(epoch, train_dice, train_run_dice[0], val_dice,val_run_dice[0]))

            self.writer.add_scalars('data/loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('data/dice', {
                'train': train_dice,
                'val': val_dice
            }, epoch)
            self.writer.add_scalars('data/run_dice', {
                'train': train_run_dice[0],
                'val': val_run_dice[0]
            }, epoch)


            self.writer.add_scalar('data/lr', optimizer.param_groups[0]['lr'], epoch)

            if self.sampler_name is not None:
                self.writer.add_scalars('data/dice_dataratio', {
                    'train': train_run_dice[0],
                    'val': val_run_dice[0],
                    'data_ratio':len(self.labeled_data_pool)/len(train_path),
                }, epoch)
            
            
            record_sample_list(
                record_csv=os.path.join(log_dir,'sample_list.csv'),
                key=str(epoch),
                sample_list=self.labeled_data_pool)
            with open(os.path.join(log_dir,'record.csv'), 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                rows = [epoch,len(self.labeled_data_pool)/len(train_path),len(self.unlabeled_data_pool)/len(train_path), \
                                    train_run_dice[0],val_run_dice[0]]
                rows.extend(train_run_dice[1][1:])
                rows.extend(val_run_dice[1][1:])
                csvwriter.writerow(rows)

            '''
            if val_loss < self.loss_threshold:
                self.loss_threshold = val_loss
            '''
            early_stopping(val_run_dice[0])

            #save
            if val_run_dice[0] > self.metrics_threshold:
                self.metrics_threshold = val_run_dice[0]

                if len(self.device.split(',')) > 1:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()

                saver = {
                    'epoch': epoch,
                    'save_dir': segnet_output_dir,
                    'state_dict': state_dict,
                    #'optimizer':optimizer.state_dict(), #TODO resume
                    #'sample_count':sample_count
                }

                file_name = 'epoch={}-train_loss={:.5f}-train_dice={:.5f}-train_run_dice={:.5f}-val_loss={:.5f}-val_dice={:.5f}-val_run_dice={:.5f}.pth'.format(
                    epoch, train_loss, train_dice, train_run_dice[0], val_loss,val_dice, val_run_dice[0])
                save_path = os.path.join(segnet_output_dir, file_name)
                print("Save as: %s" % file_name)

                torch.save(saver, save_path)

            #early stopping
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self.writer.close()
        dfs_remove_weight(output_dir, 3)
        print(f'total sample time is {total_sample_time}')
        print(f'sample time of every iter is {total_sample_time/self.sample_times}')


    def _get_data_loader(self, data_path=[], data_type='train', repeat_factor=1.0):

        assert len(data_path) != 0

        if data_type == 'train':
            data_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale,self.channels),
                Get_ROI(pad_flag=False) if self.get_roi else transforms.Lambda(lambda x: x),
                CropResize(dim=self.input_shape,
                        num_class=self.num_classes,
                        crop=self.crop,
                        channels=self.channels),
                # RandomErase2D(scale_flag=False),
                # RandomZoom2D(), # bug for normalized MR image #TODO
                RandomDistort2D(),
                RandomRotate2D(),
                RandomFlip2D(mode='v'),
                # # RandomAdjust2D(),
                RandomNoise2D(),
                To_Tensor(num_class=self.num_classes,channels=self.channels)
            ])

        elif data_type == 'val':
            data_transformer = transforms.Compose([
                Trunc_and_Normalize(self.scale,self.channels),
                Get_ROI(pad_flag=False) if self.get_roi else transforms.Lambda(lambda x: x),
                CropResize(dim=self.input_shape,
                        num_class=self.num_classes,
                        crop=self.crop,
                        channels=self.channels),
                To_Tensor(num_class=self.num_classes,channels=self.channels)
            ])
            
        dataset = DataGenerator(data_path,
                                roi_number=self.roi_number,
                                num_class=self.num_classes,
                                transform=data_transformer,
                                repeat_factor=repeat_factor)

        data_loader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=data_type == 'train',
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=self.sampler_name == 'loss_predictor')

        return data_loader
    

    def _random_sampling(self, sample_pool=[]):
        assert len(sample_pool) != 0
        samples = random.sample(sample_pool,k=int(len(sample_pool) * self.init_percent))
        return samples


    def _update_unlabeled_data_pool(self, labeled_data=[]):
        if len(labeled_data) != 0:
            for sample in labeled_data:
                self.unlabeled_data_pool.remove(sample)


    def _get_samples_per_epoch(self, N_sample, sample_times, sample_mode):
        
        init_sample = int(self.init_percent * N_sample)
        sample_strategy = np.zeros((sample_times+1,))
        sample_strategy[0] = init_sample

        n_sample_num = int(self.max_percent * N_sample - init_sample)
        first_sample = min(int(n_sample_num/(sample_times//2)),init_sample)
        sample_strategy[1] = first_sample

        if sample_mode == "uniform":
            d = n_sample_num // sample_times  
            for i in range(1,sample_times + 1):
                sample_strategy[i] = d

        elif sample_mode == "linear":
            d = int(2 * (sample_times * first_sample - n_sample_num) /
                    (sample_times * (sample_times - 1)))
            # print(d)
            for i in range(2,sample_times + 1):
                sample_strategy[i] = max(sample_strategy[i - 1] - d, 0)

        elif sample_mode == "convex":
            for sample_times in reversed(range(1, sample_times)):
                if 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1)) > (sample_times-1)**2/first_sample:
                    break
            d = (9*(sample_times*first_sample - n_sample_num)**2)/(4*(sample_times-1)**3)
            for t,i in enumerate(range(2, sample_times + 1)):
                sample_strategy[i] = int(first_sample - (d*t)**0.5)

        elif sample_mode == "square":
            for sample_times in reversed(range(1, sample_times)):
                if 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1)) > (sample_times-1)**2/first_sample :
                    break
            d = 6*(sample_times*first_sample - n_sample_num)/(sample_times*(sample_times-1)*(2*sample_times-1))
            for t,i in enumerate(range(2, sample_times + 1)):
                sample_strategy[i] = max(int(first_sample - d*t**2),0)

        return sample_strategy


    def _train_on_epoch(self, epoch, net, criterion, optimizer, train_loader, scaler):

        net.train()

        train_loss = AverageMeter()
        train_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)

        for step, sample in enumerate(train_loader):

            data = sample['image']
            target = sample['label']

            data = data.cuda()
            target = target.cuda()

            with autocast(self.use_fp16):
                output = net(data)
                loss = criterion(output, target)

                if isinstance(output, tuple):
                    output = output[0]

            optimizer.zero_grad()
            if self.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            output = output.float()
            loss = loss.float()

            # measure dice and record loss
            dice = compute_dice(output.detach(), target)
            train_loss.update(loss.item(), data.size(0))
            train_dice.update(dice, data.size(0))

            # measure run dice
            output = torch.argmax(torch.softmax(output, dim=1),
                                  1).detach().cpu().numpy()  #N*H*W
            target = torch.argmax(target, 1).detach().cpu().numpy()
            run_dice.update_matrix(target, output)

            torch.cuda.empty_cache()

            if self.global_step % 2 == 0:
                rundice, dice_list = run_dice.compute_dice()
                print("Category Dice: ", dice_list)
                print(
                    'epoch:{},step:{},train_loss:{:.5f},train_dice:{:.5f},train_run_dice:{:.5f},lr:{:.5f}'
                    .format(epoch, step, loss.item(), dice, rundice,
                            optimizer.param_groups[0]['lr']))
                # run_dice.init_op()
                self.writer.add_scalars('data/train_loss_dice', {
                    'train_loss': loss.item(),
                    'train_dice': dice,
                }, self.global_step)

            self.global_step += 1

        # return train_loss.avg,run_dice.compute_dice()[0]
        return train_loss.avg, train_dice.avg, run_dice.compute_dice()


    def _val_on_epoch(self,
                      epoch,
                      net,
                      criterion,
                      val_loader):

        net.eval()

        val_loss = AverageMeter()
        val_dice = AverageMeter()

        from metrics import RunningDice
        run_dice = RunningDice(labels=range(self.num_classes), ignore_label=-1)

        with torch.no_grad():
            for step, sample in enumerate(val_loader):
                data = sample['image']
                target = sample['label']

                data = data.cuda()
                target = target.cuda()
                with autocast(self.use_fp16):
                    output = net(data)
                    loss = criterion(output, target)
                    
                    if isinstance(output, tuple):
                        output = output[0]
               
                output = output.float()
                loss = loss.float()

                # measure dice and record loss
                dice = compute_dice(output.detach(), target)
                val_loss.update(loss.item(), data.size(0))
                val_dice.update(dice, data.size(0))

                # measure run dice
                output = torch.argmax(torch.softmax(output, dim=1),
                                      1).detach().cpu().numpy()  #N*H*W
                target = torch.argmax(target, 1).detach().cpu().numpy()
                run_dice.update_matrix(target, output)

                torch.cuda.empty_cache()

                if step % 2 == 0:
                    rundice, dice_list = run_dice.compute_dice()
                    print("Category Dice: ", dice_list)
                    print(
                        'epoch:{},step:{},val_loss:{:.5f},val_dice:{:.5f},val_run_dice:{:.5f}'
                        .format(epoch, step, loss.item(), dice,
                                rundice))
                    # run_dice.init_op()

        # return val_loss.avg,run_dice.compute_dice()[0]
        return val_loss.avg, val_dice.avg, run_dice.compute_dice()


    def _get_seg_net(self, net_name):

        if net_name == 'unet':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                if self.encoder_name in ['resnet50_dropout','resnet50_naive']:
                    from model.unet import unet
                    net = unet(net_name,
                        encoder_name=self.encoder_name,
                        in_channels=self.channels,
                        classes=self.num_classes,
                        aux_losspredictor=self.sampler_name == 'loss_predictor'
                    )
                else:
                    import segmentation_models_pytorch as smp
                    net = smp.Unet(encoder_name=self.encoder_name,
                                encoder_weights=None
                                if not self.ex_pre_trained else 'imagenet',
                                in_channels=self.channels,
                                classes=self.num_classes)
        elif net_name == 'unet++':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                       encoder_weights=None if
                                       not self.ex_pre_trained else 'imagenet',
                                       in_channels=self.channels,
                                       classes=self.num_classes)

        elif net_name == 'FPN':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.FPN(encoder_name=self.encoder_name,
                              encoder_weights=None
                              if not self.ex_pre_trained else 'imagenet',
                              in_channels=self.channels,
                              classes=self.num_classes)

        elif net_name == 'deeplabv3+':
            if self.encoder_name is None:
                raise ValueError("encoder name must not be 'None'!")
            else:
                import segmentation_models_pytorch as smp
                net = smp.DeepLabV3Plus(
                    encoder_name=self.encoder_name,
                    encoder_weights=None
                    if not self.ex_pre_trained else 'imagenet',
                    in_channels=self.channels,
                    classes=self.num_classes)

        return net


    def _get_sampler(self, sampler_name):

        if sampler_name == 'random':
            from strategy.base_strategy import random_sampler
            sampler = random_sampler

        elif sampler_name == 'entropy':
            from strategy.base_strategy import entropy_sampler
            sampler = entropy_sampler

        elif sampler_name == 'leastconfidence':
            from strategy.base_strategy import leastconfidence_sampler
            sampler = leastconfidence_sampler

        elif sampler_name == 'varratio':
            from strategy.base_strategy import varratio_sampler
            sampler = varratio_sampler

        elif sampler_name == 'margin':
            from strategy.base_strategy import margin_sampler
            sampler = margin_sampler
        
        elif sampler_name == 'kmeans':
            from strategy.base_strategy import kmeans_sampler
            sampler = kmeans_sampler
        
        elif sampler_name == 'ceal_entropy':
            from strategy.base_strategy import ceal_entropy_sampler
            sampler = ceal_entropy_sampler
        
        elif sampler_name == 'kcenter_pca':
            from strategy.base_strategy import kcenter_pca_sampler
            sampler = kcenter_pca_sampler

        elif sampler_name == 'bayesian':
            from strategy.base_strategy import bayesian_sampler
            sampler = bayesian_sampler

        elif sampler_name == 'loss_predictor':
            from strategy.base_strategy import lp_sampler
            sampler = lp_sampler

        elif sampler_name == 'entropy_kmeans':
            from strategy.base_strategy import entropy_kmeans_sampler
            sampler = entropy_kmeans_sampler
        

        return sampler
        

    def _get_seg_loss(self, loss_fun, class_weight=None):
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)

        if loss_fun == 'Cross_Entropy':
            from loss.cross_entropy import CrossentropyLoss
            loss = CrossentropyLoss(weight=class_weight)

        elif loss_fun == 'TopKLoss':
            from loss.cross_entropy import TopKLoss
            loss = TopKLoss(weight=class_weight, k=self.topk)

        elif loss_fun == 'CELabelSmoothingPlusDice':
            from loss.combine_loss import CELabelSmoothingPlusDice
            loss = CELabelSmoothingPlusDice(smoothing=0.1,
                                            weight=class_weight,
                                            ignore_index=0)

        elif loss_fun == 'OHEM':
            from loss.cross_entropy import OhemCELoss
            loss = OhemCELoss(thresh=0.7)

        elif loss_fun == 'DiceLoss':
            from loss.dice_loss import DiceLoss
            loss = DiceLoss(weight=class_weight, ignore_index=0, p=1)

        elif loss_fun == 'CEPlusDice':
            from loss.combine_loss import CEPlusDice
            loss = CEPlusDice(weight=class_weight, ignore_index=0)

        elif loss_fun == 'CEPlusLPL':
            from loss.lp_loss import CEPlusLPL
            loss = CEPlusLPL(reduction='mean', alpha=1.0, margin=1.0)
        
        return loss


    def _get_optimizer(self, optimizer, net, lr):
        """
        Build optimizer, set weight decay of normalization to 0 by default.
        """
        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin

        def set_weight_decay(model, skip_list=(), skip_keywords=()):
            has_decay = []
            no_decay = []

            for name, param in model.named_parameters():
                # check what will happen if we do not set no_weight_decay
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    no_decay.append(param)
                    # print(f"{name} has no weight decay")
                else:
                    has_decay.append(param)
            return [{
                'params': has_decay
            }, {
                'params': no_decay,
                'weight_decay': 0.
            }]

        skip = {}
        skip_keywords = {}
        if hasattr(net, 'no_weight_decay'):
            skip = net.no_weight_decay()
        if hasattr(net, 'no_weight_decay_keywords'):
            skip_keywords = net.no_weight_decay_keywords()
        parameters = set_weight_decay(net, skip, skip_keywords)

        opt_lower = optimizer.lower()
        optimizer = None
        if opt_lower == 'sgd':
            optimizer = torch.optim.SGD(parameters,
                                        momentum=self.momentum,
                                        nesterov=True,
                                        lr=lr,
                                        weight_decay=self.weight_decay)
        elif opt_lower == 'adamw':
            optimizer = torch.optim.AdamW(parameters,
                                          eps=1e-8,
                                          betas=(0.9, 0.999),
                                          lr=lr,
                                          weight_decay=self.weight_decay)
        elif opt_lower == 'adam':
            optimizer = torch.optim.Adam(parameters,
                                         lr=lr,
                                         weight_decay=self.weight_decay)

        return optimizer


    def _get_lr_scheduler(self, lr_scheduler, optimizer):
        if lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, verbose=True)
        elif lr_scheduler == 'CustomScheduler':
            from custom_scheduler import CustomScheduler
            lr_scheduler = CustomScheduler(
                optimizer=optimizer,
                max_lr=self.lr,
                min_lr=1e-6,
                lr_warmup_steps=self.warmup_epoch,
                lr_decay_steps=self.n_epoch,
                lr_decay_style='cosine',
                start_wd=self.weight_decay,
                end_wd=self.weight_decay,
                wd_incr_style='constant',
                wd_incr_steps=self.n_epoch
            )
        elif lr_scheduler == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.milestones, gamma=self.gamma)
        elif lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_epoch, eta_min=1e-6)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 20, T_mult=2)
        return lr_scheduler


    def _get_pre_trained_seg_net(self, weight_path, ckpt_point=True):
        checkpoint = torch.load(weight_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        if ckpt_point:
            self.start_epoch = checkpoint['epoch'] + 1
            self.metrics_threshold = eval(
                self.weight_path.split('=')[-2].split('-')[0])



# computing tools


class AverageMeter(object):
    '''
  Computes and stores the average and current value
  '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def binary_dice(predict, target, smooth=1e-5, reduction='mean'):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1e-5
        predict: A numpy array of shape [N, *]
        target: A numpy array of shape same with predict
    Returns:
        DSC numpy array according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    assert predict.shape[0] == target.shape[
        0], "predict & target batch size don't match"
    predict = predict.reshape(predict.shape[0], -1)  #N，H*W
    target = target.reshape(target.shape[0], -1)  #N，H*W

    inter = np.sum(np.multiply(predict, target), axis=1)  #N
    union = np.sum(predict + target, axis=1)  #N

    dice = (2 * inter + smooth) / (union + smooth)  #N

    if reduction == 'mean':
        # nan mean
        dice_index = dice != 1.0
        dice = dice[dice_index]
        return dice.mean()
    else:
        return dice  #N


def compute_dice(predict, target, ignore_index=0, reduction='mean'):
    """
    Compute dice
    Args:
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        ignore_index: class index to ignore
    Return:
        mean dice over the batch
    """
    N, num_classes, _, _ = predict.size()
    assert predict.shape == target.shape, 'predict & target shape do not match'
    predict = F.softmax(predict, dim=1)

    predict = torch.argmax(predict, dim=1).detach().cpu().numpy()  #N*H*W
    target = torch.argmax(target, dim=1).detach().cpu().numpy()  #N*H*W

    if reduction == 'mean':
        dice_array = -1.0 * np.ones((num_classes, ), dtype=np.float32)  #C
    else:
        dice_array = -1.0 * np.ones((num_classes, N), dtype=np.float32)  #CN

    for i in range(num_classes):
        if i != ignore_index:
            if i not in predict and i not in target:
                continue
            dice = binary_dice((predict == i).astype(np.float32),
                               (target == i).astype(np.float32),
                               reduction=reduction)
            dice_array[i] = np.round(dice, 4)

    if reduction == 'mean':
        dice_array = np.where(dice_array == -1.0, np.nan, dice_array)
        return np.nanmean(dice_array[1:])
    else:
        dice_array = np.where(dice_array == -1.0, 1.0,
                              dice_array).transpose(1, 0)  #CN -> NC
        return dice_array



class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""
    def __init__(self,
                 patience=10,
                 verbose=True,
                 delta=0,
                 monitor='val_loss',
                 op_type='min'):
        """
        Args:
            patience (int): How long to wait after last time performance improved.
                            Default: 10
            verbose (bool): If True, prints a message for each performance improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            monitor (str): Monitored variable.
                            Default: 'val_loss'
            op_type (str): 'min' or 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type

        if self.op_type == 'min':
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):

        score = -val_score if self.op_type == 'min' else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0

    def print_and_update(self, val_score):
        '''print_message when validation score decrease.'''
        if self.verbose:
            print(
                self.monitor,
                f'optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...'
            )
        self.val_score_min = val_score