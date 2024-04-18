import torch
from torch.cuda.amp import autocast as autocast
from sklearn.cluster import KMeans

import numpy as np
import random

import os
import h5py

from sklearn.decomposition import PCA
from tqdm import tqdm
import copy
import shutil
import time

def random_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    labeled_path = random.sample(unlabeled_data_pool, k=int(sample_nums))

    return labeled_path

###################### Uncertain-based

def entropy_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    # s_time = time.time()
    # total_inf_time = 0
    # total_loader_time = 0

    seg_net.eval()
    output_list = []
    print("******* Start predicting unlabeled data *******")

    # d_time = time.time()
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            # total_loader_time += (time.time() - d_time)
            # print('NO.%d step data loader time:%.4f' % (step,(time.time() - d_time)))
            # inf_time = time.time()

            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                # total_inf_time += (time.time() - inf_time)
                # step_time = time.time() 

                # output = torch.softmax(output, dim=1).detach().cpu()
                output = torch.softmax(output, dim=1).detach()
                output = output.mean(dim=(2,3)) # NCHW -> NC
                log_probs = torch.log(output)
                uncertainties = -(output*log_probs).sum(1).cpu() # -q * log(q)

                # print('NO.%d step time:%.4f' % (step,(time.time() - step_time)))

            output_list.extend(uncertainties)
            # d_time = time.time()

    # print('compute only time:%.4f' % (time.time() - s_time))

    score_arr = np.array(output_list)
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    # print('sample time:%.4f' % (time.time() - s_time))
    # print('total inf time:%.4f' % total_inf_time)
    # print('total data loader time:%.4f' % total_loader_time)
    return labeled_path



def leastconfidence_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    seg_net.eval()

    output_list = []
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
                output = torch.softmax(output, dim=1).detach()
                output = output.mean(dim=(2,3)) # NCHW -> NC
                probs = -output.max(1)[0].cpu()  # - max(p(y|x))
                uncertainties = probs.tolist()
            output_list.extend(uncertainties)
    
    score_arr = np.array(output_list)
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path



def varratio_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    seg_net.eval()

    output_list = []
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
                output = torch.softmax(output, dim=1).detach()
                output = output.mean(dim=(2,3)) # NCHW -> NC
                probs = output.max(1)[0].cpu() # max(p(y|x))
                uncertainties = (1.0 - probs).tolist()
            output_list.extend(uncertainties)
    
    score_arr = np.array(output_list)
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path


def margin_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    seg_net.eval()

    output_list = []
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
                output = torch.softmax(output, dim=1).detach() # NCHW
                output = output.mean(dim=(2,3)).cpu()  # NCHW -> NC
                probs_sorted, idxs = output.sort(descending=True)
                uncertainties = probs_sorted[:,1] - probs_sorted[:, 0]
            output_list.extend(uncertainties)

    score_arr = np.array(output_list)
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path


def bayesian_sampler(seg_net, unlabeled_data_pool, sample_loader, sample_nums, n_drop=10):

    seg_net.eval()

    
    probs = []
    print("******* Start predicting unlabeled data *******")
    for i in range(n_drop):
        output_list = []
        with torch.no_grad():
            for step, sample in enumerate(sample_loader):
                data = sample['image']
                data = data.cuda()

                with autocast(True):
                    output = seg_net(data)#NCHW
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    output = torch.softmax(output, dim=1).detach() # NCHW
                    output = output.mean(dim=(2,3)).cpu() # NCHW -> NC
                output_list.append(output)
        
        probs.append(torch.cat(output_list, dim=0))
    
    probs = torch.stack(probs,dim=0) # n_drop, N_, C
    pb = probs.mean(0) # N_, C
    entropy1 = (-pb*torch.log(pb)).sum(1) # N_
    entropy2 = (-probs*torch.log(probs)).sum(2).mean(0) # N_
    uncertainties = entropy2 - entropy1 # N_

    K = int(sample_nums)
    indices = uncertainties.sort()[1][:K]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path


###################### representation-based

def kmeans_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    seg_net.eval()
    representations = []
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def hook_fn_forward(module, input, output):
        # print(output[-1].size())
        representations.append(avgpool(output[-1]).detach().cpu().numpy().squeeze(axis=(2,3))) #NC
    # for smp model zoo
    handle = seg_net.encoder.register_forward_hook(hook_fn_forward)
    
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
    handle.remove()
    representation_array = np.concatenate(representations,axis=0)

    K = int(sample_nums)
    cluster_learner = KMeans(n_clusters=K, random_state=0)
    cluster_learner.fit(representation_array)
    
    cluster_idxs = cluster_learner.predict(representation_array)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (representation_array - centers)**2
    dis = dis.sum(axis=1)
    indices = np.array([np.arange(representation_array.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(K)])

    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path



def kcenter_pca_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    seg_net.eval()
    representations = []
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def hook_fn_forward(module, input, output):
        # print(output[-1].size())
        representations.append(avgpool(output[-1]).detach().cpu().numpy().squeeze(axis=(2,3))) #NC
    # for smp model zoo
    handle = seg_net.encoder.register_forward_hook(hook_fn_forward)
    
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
    handle.remove()
    embeddings = np.concatenate(representations,axis=0) #N,C
    labeled_idxs = np.ones((embeddings.shape[0],))
    labeled_idxs[:len(unlabeled_data_pool)] = 0
    labeled_idxs = np.where(labeled_idxs == 1, True, False)

    raw_labeled_idxs = copy.deepcopy(labeled_idxs)

     #downsampling embeddings if feature dim > 50
    embeddings = (embeddings - embeddings.mean()) / embeddings.std()

    embeddings = embeddings.astype(np.float16)
    try:
        if embeddings.shape[1] > 50:
            # pca = PCA(n_components=min(128,min(*embeddings.shape)))
            pca = PCA(n_components=50)
            embeddings = pca.fit_transform(embeddings)
        embeddings = embeddings.astype(np.float16)
    except:
        print('numpy.linalg.LinAlgError: SVD did not converge')
        embeddings = embeddings.astype(np.float16)
    
    print(embeddings.shape)
    dist_mat = np.matmul(embeddings, embeddings.transpose())
    print(dist_mat.shape)
    sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)

    mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

    for i in tqdm(range(int(sample_nums)), ncols=100):
        mat_min = mat.min(axis=1)
        q_idx_ = mat_min.argmax()
        q_idx = np.arange(embeddings.shape[0])[~labeled_idxs][q_idx_]
        labeled_idxs[q_idx] = True
        mat = np.delete(mat, q_idx_, 0)
        mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
        
    indices = np.arange(embeddings.shape[0])[(raw_labeled_idxs ^ labeled_idxs)] # value in indices must < len(unlabeled_data_pool) 
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path


###################### ceal: + semi

def store_image_label(save_dir, data_path, image, label):
    for i, item in enumerate(data_path):
        save_path = os.path.join(save_dir, os.path.basename(item))
        hdf5_file = h5py.File(save_path, 'w')
        hdf5_file.create_dataset('image', data=image[i].astype(np.float32))
        hdf5_file.create_dataset('label', data=label[i].astype(np.uint8))
        hdf5_file.close()

def ceal_entropy_sampler(seg_net, unlabeled_data_pool, sample_loader, sample_nums, semi_save_dir=None,delta=5e-5):

    seg_net.eval()

    if os.path.exists(semi_save_dir):
        shutil.rmtree(semi_save_dir)
        os.makedirs(semi_save_dir)
    else:
        os.makedirs(semi_save_dir)

    output_list = []
    print("******* Start predicting unlabeled data *******")
    index = 0
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
                output_prob = torch.softmax(output, dim=1).detach()
                output_prob = output_prob.mean(dim=(2,3)) # NCHW -> NC
                log_probs = torch.log(output_prob)
                uncertainties = -(output_prob*log_probs).sum(1).cpu() # -q * log(q)
            output_list.extend(uncertainties)

            # save as hdf5
            data_size = data.size(0) # N
            data_numpy = data.detach().cpu().numpy().squeeze() #NCHW, if C=1, NHW
            output_numpy = torch.argmax(torch.softmax(output, dim=1),1).detach().cpu().numpy() #NHW
            data_path = unlabeled_data_pool[index:index + data_size]
            store_image_label(semi_save_dir,data_path,data_numpy,output_numpy)
            
            index += data_size
    
    # entropy sampler
    score_arr = np.array(output_list) # 1D sequence
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    # semi sampler
    high_confident_idx = np.where(score_arr < delta)[0]
    semi_data_name = [os.path.basename(unlabeled_data_pool[i]) for i in high_confident_idx]
    semi_data_path = [os.path.join(semi_save_dir,case) for case in semi_data_name]

    # remove extra hdf5 file
    for item in os.scandir(semi_save_dir):
        if os.path.basename(item.path) not in semi_data_name:
            os.remove(item.path)

    return labeled_path, semi_data_path



###################### loss predictor

def lp_sampler(seg_net, unlabeled_data_pool, sample_loader,sample_nums):

    if not seg_net.detach_flag:
        seg_net.detach_flag = True
        
    seg_net.eval()

    output_list = []
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                assert isinstance(output, tuple)
                output = output[1] #N,1
                
                uncertainties = output.view(output.size(0)).detach().cpu()
            output_list.extend(uncertainties)
    
    score_arr = np.array(output_list)
    K = int(sample_nums)
    indices = np.argpartition(score_arr, -K)[-K:]
    labeled_path = [unlabeled_data_pool[i] for i in indices]

    return labeled_path



###################### entropy + kmeans

def entropy_kmeans_sampler(seg_net, unlabeled_data_pool, sample_loader, sample_nums):

    seg_net.eval()

    representations = []
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def hook_fn_forward(module, input, output):
        # print(output[-1].size())
        representations.append(avgpool(output[-1]).detach().cpu().numpy().squeeze(axis=(2,3))) #NC
    # for smp model zoo
    handle = seg_net.encoder.register_forward_hook(hook_fn_forward)

    output_list = []
    print("******* Start predicting unlabeled data *******")
    with torch.no_grad():
        for step, sample in enumerate(sample_loader):
            data = sample['image']
            data = data.cuda()

            with autocast(True):
                output = seg_net(data)#NCHW
                if isinstance(output, tuple):
                    output = output[0]
                
                output = torch.softmax(output, dim=1).detach()
                output = output.mean(dim=(2,3)) # NCHW -> NC
                log_probs = torch.log(output)
                uncertainties = -(output*log_probs).sum(1).cpu() # -q * log(q)
            output_list.extend(uncertainties)
    
    handle.remove()

    score_arr = np.array(output_list)

    max_K = min(int(sample_nums*20),len(unlabeled_data_pool))
    # get the indices of the top-k largest values
    extend_indices = np.argpartition(score_arr, -max_K)[-max_K:]
    extend_labeled_path = [unlabeled_data_pool[i] for i in extend_indices]
    representation_array = np.concatenate(representations,axis=0)[extend_indices] #K * C

    K = int(sample_nums)
    cluster_learner = KMeans(n_clusters=K, random_state=0)
    cluster_learner.fit(representation_array)
    
    cluster_idxs = cluster_learner.predict(representation_array)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (representation_array - centers)**2
    dis = dis.sum(axis=1)
    indices = np.array([np.arange(representation_array.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(K)])

    labeled_path = [extend_labeled_path[i] for i in indices]

    return labeled_path