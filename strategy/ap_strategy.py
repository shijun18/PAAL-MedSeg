import torch
from torch.cuda.amp import autocast as autocast
from sklearn.cluster import KMeans


import numpy as np
import math


def acc_predictor(seg_net,predictor,unlabeled_data_pool,sample_loader,sample_nums,sample_weight=None,al_mode='single',score_type='mean'):

    seg_net.eval()
    predictor.eval()

    predictor_scores = []
    representations = []
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def hook_fn_forward(module, input, output):
        # print(output[-1].size())
        representations.append(avgpool(output[-1]).detach().cpu().numpy().squeeze(axis=(2,3))) #NC
    
    if not al_mode == 'ap':
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

                predictor_data = torch.stack([torch.cat((i, j), dim=0) for i, j in zip(data.clone().detach(),torch.softmax(output.clone().detach(), dim=1))],dim=0)  #N,C+1,H,W
                predictor_output = predictor(predictor_data) #NC
                scores = list(compute_score(pred_out=predictor_output,weights=sample_weight,score_type=score_type)) #N
                predictor_scores.extend(scores)
    
    if not al_mode == 'ap':
        handle.remove()
    
    score_arr = np.array(predictor_scores)
    # print(score_arr.shape)
    # print(type(representations))
    if al_mode == 'ap':
        K = int(sample_nums)
        # get the indices of the top-k smallest values
        indices = np.argpartition(score_arr, K)[:K]
        labeled_path = [unlabeled_data_pool[i] for i in indices]

    else:
        # over-sampling for large scale dataset
        # max_K = min(int(sample_nums*4),len(unlabeled_data_pool))
        # get the indices of the top-k smallest values
        # extend_indices = np.argpartition(score_arr, max_K)[:max_K]
        # extend_labeled_path = [unlabeled_data_pool[i] for i in extend_indices]
        # representation_array = np.concatenate(representations,axis=0)[extend_indices] #K * C
        # weights_array = score_arr[extend_indices]

        extend_labeled_path = unlabeled_data_pool
        representation_array = np.concatenate(representations,axis=0)
        weights_array = score_arr

        if al_mode == 'ap+wps': #TODO kcenter method
            # sim_matrix = similarity_cal(representation_array) #max_K * max_K
            # cluster_result = union_find(sim_matrix,threshold=0.9) # max_K, can be set to other values
            kmeans = KMeans(n_clusters=int(math.log(sample_nums*4,2) + 1), random_state=0)
            kmeans.fit(representation_array)
            cluster_result = kmeans.labels_
            print('Number of Cluster Centers = %d'%len(set(cluster_result)))
            cluster_dict = merge_class(cluster_result,index_list=extend_labeled_path,weights=weights_array)
            labeled_path = pollsampling(cluster_dict,sample_nums=int(sample_nums))

    return labeled_path



def compute_score(pred_out,weights=None,score_type='mean'):
    pred_out = pred_out.detach().cpu().numpy()
    if weights is not None:
        pred_out = pred_out * np.array(weights)
    if score_type == 'mean':
        scores = pred_out.mean(axis=1)
    elif score_type == 'log_mean':
        scores = np.log(pred_out).mean(axis=1)

    return scores


def pollsampling(cluster_dict,sample_nums):
    sample_list = []
    keys = list(cluster_dict.keys())
    count = 0
    while len(sample_list) < sample_nums:
        key = keys[int(count % len(keys))]
        if len(cluster_dict[key]) != 0:
            sample_list.append(cluster_dict[key].pop())
        count += 1
    return sample_list


def similarity_cal(vector):
    from tqdm import tqdm
    N,_ = vector.shape
    sim_matrix = np.zeros((N,N),dtype=np.float32)
    for i in tqdm(range(N)):
        for j in range(i,N): 
            sim_matrix[i][j] = cosine_similarity(vector[i],vector[j])
    index_low = np.tril_indices(N)
    sim_matrix[index_low] = np.tril(sim_matrix.T)[index_low]
    sim_matrix = (sim_matrix + 1.0) / 2.0 # normalize to 0~1
    return sim_matrix


def cosine_similarity(vector1, vector2):  
    dot_product = np.dot(vector1, vector2)  
    norm1 = np.linalg.norm(vector1)  
    norm2 = np.linalg.norm(vector2)  
    return dot_product / (norm1 * norm2) 



def union_find(data,threshold=0.7):
    def merge(id1, id2, fa):
        fa,fa1 = findfa(id1,fa)
        fa,fa2 = findfa(id2,fa)
        if fa1 != fa2:
            fa[fa1] = fa2
        return fa

    def findfa(id,fa):
        if fa[id] == -1:
            return fa,id
        fa,fa[id] = findfa(fa[id],fa)
        return fa,fa[id]
    
    col,row = data.shape
    fa = [-1 for i in range(row)]

    for i in range(row):
        for j in range(col):
            if data[i][j] > threshold:
                fa = merge(i, j, fa)
                # print(i,j,fa)

    result = []
    for i in range(row):
        if findfa(i,fa)[1] == i:
            result.append(i)
        else:
            result.append(fa[i])

    return result

def merge_class(class_result,index_list,weights=None):
    assert len(class_result) == len(index_list)
    class_set = list(set(class_result))
    cluster_dict = {}
    for i in range(len(class_set)):
        cluster_dict[str(i)] = []
    for index,item in enumerate(class_result):
        key = class_set.index(item)
        cluster_dict[str(key)].append([index_list[index],weights[index]])

    for key in cluster_dict.keys():
        cluster_dict[key].sort(key=lambda x: x[1], reverse=True)  
        # print(cluster_dict[key])
        cluster_dict[key] = [case[0] for case in cluster_dict[key]]
    return cluster_dict