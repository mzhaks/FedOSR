import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy
import platform
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, random_split
from collections import Counter

def get_dataloaders(client_num, data_root, seed, param={
    'Known_class': 5, 'unKnown_class': 3, 'Batchsize': 8, 'dirichlet': 1.0}):
    
    known_class = param['Known_class']
    unknown_class = param['unKnown_class']
    batchsize = param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    total_class = 10  # CIFAR-10 has 10 classes
    
    assert known_class + unknown_class <= total_class
    
    np.random.seed(seed)
    state = np.random.get_state()
    class_list = np.arange(total_class)
    np.random.shuffle(class_list)
    known_class_list = class_list[:known_class]
    unknown_class_list = class_list[known_class:known_class + unknown_class]
    
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    full_testset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # Define split sizes (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size

    # Split training dataset into train and validation sets
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # Accessing the data
    trainx, trainy = zip(*[(x, y) for x, y in trainset])
    valx, valy = zip(*[(x, y) for x, y in valset])
    testx, testy = zip(*[(x, y) for x, y in full_testset])
    d = Counter(trainy[:,0])
    d_s = sorted(d.items(),key=lambda x:x[1],reverse=True)
    print('The number of class each',d_s)    
    print('The numbers of Training, Val, Test sets, {}, {}, {}'.format(len(trainx), len(valx), len(testx)))

    #relabel dataset
    knowndict={}
    unknowndict={}
    for i in range(len(known_class_list)):
        knowndict[known_class_list[i]]=i
    for j in range(len(unknown_class_list)):
        unknowndict[unknown_class_list[j]]=j+len(known_class_list)
    print(knowndict, unknowndict)

    trainy = np.array(trainy)
    valy = np.array(valy)        
    testy = np.array(testy)        
    copytrainy=copy.deepcopy(trainy)
    copyvaly=copy.deepcopy(valy)
    copytesty=copy.deepcopy(testy)  
    for i in range(len(known_class_list)):
        #修改已知类标签
        trainy[copytrainy==known_class_list[i]]=knowndict[known_class_list[i]] 
        valy[copyvaly==known_class_list[i]]=knowndict[known_class_list[i]]             
        testy[copytesty==known_class_list[i]]=knowndict[known_class_list[i]]
    for j in range(len(unknown_class_list)):
        #修改未知类标签
        trainy[copytrainy==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]
        valy[copyvaly==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]             
        testy[copytesty==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]   
    origin_known_list=known_class_list
    origin_unknown_list=unknown_class_list
    new_known_list=np.arange(known_class)
    new_unknown_list=np.arange(known_class, known_class+len(unknown_class_list))
    print(origin_known_list, new_known_list, origin_unknown_list, new_unknown_list)
     
    #获取已知类的index 便于索引
    train_data_known_index=[]
    val_data_known_index=[]        
    test_data_known_index=[]
    for item in new_known_list:
        index=np.where(trainy==item)
        index=list(index[0])
        train_data_known_index=train_data_known_index+index
        index=np.where(valy==item)
        index=list(index[0])
        val_data_known_index=val_data_known_index+index            
        index=np.where(testy==item)
        index=list(index[0])
        test_data_known_index=test_data_known_index+index
        
    #获得未知类的index
    train_data_index_perm=np.arange(len(trainy))
    train_data_unknown_index=np.setdiff1d(train_data_index_perm,train_data_known_index)
    val_data_index_perm=np.arange(len(valy))
    val_data_unknown_index=np.setdiff1d(val_data_index_perm,val_data_known_index)        
    test_data_index_perm=np.arange(len(testy))
    test_data_unknown_index=np.setdiff1d(test_data_index_perm,test_data_known_index)
    print('Known and Unknow in Train:',len(train_data_known_index),len(train_data_unknown_index))
    print('Known and Unknow in Val:',len(val_data_known_index), len(val_data_unknown_index))
    print('Known and Unknow in Test:',len(test_data_known_index), len(test_data_unknown_index))


if __name__=='__main__':
    get_dataloaders(10, './data', 47)