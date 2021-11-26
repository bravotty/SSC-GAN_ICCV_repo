import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from itertools import combinations
# from nt_xent import NTXentLoss

cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to('cuda')
cosim0 = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to('cuda')
consistency_loss = nn.L1Loss(reduction='elementwise_mean').cuda()

def loss_hinge_dis_real(dis_real1, dis_real2, weight):
  avg_loss = F.relu(1. - dis_real1)*weight[:,0] + F.relu(1. - dis_real2)*weight[:,1]
  loss_real = torch.mean(avg_loss)
  return loss_real

def get_correlated_mask(bs):
    diag = np.eye(2 * bs)
    l1 = np.eye((2 * bs), 2 * bs, k=-bs)
    l2 = np.eye((2 * bs), 2 * bs, k=bs)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.uint8)
    return mask.to('cuda')

def cosim_own(I1, I2):
  c = 1e-6 # prevent divid 0
  InnerPro = torch.sum(I1*I2,1,keepdim=True) # N,1,H,W
  len1 = torch.norm(I1, p=2,dim=1,keepdim=True) # ||x1||
  len2 = torch.norm(I2, p=2,dim=1,keepdim=True) # ||x2||

  divisor = len1*len2 # ||x1||*||x2||, N,1,H,W
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*c # prevent divids 0
  cosA = torch.sum(InnerPro/divisor,1) # N,H,W

  return cosA

def triplet_loss_simclr(emb, y, margin, typen, size=0):
    with torch.no_grad():
        # triplets = get_triplets(emb, y, margin, typen, size)
        triplets = get_triplets_cos(emb, y, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]
    ap_D = (f_A - f_P).pow(2).sum(1).pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1).pow(.5)

    loss = torch.log(torch.sum(torch.exp(ap_D))/(torch.sum(torch.exp(ap_D))+torch.sum(torch.exp(an_D))))

    return torch.mean(loss)

def triplet_loss_cos(emb, y, typen, size=0):
    with torch.no_grad():
        triplets = get_triplets_cos(emb, y, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_cos = cosim(f_A, f_P)  # .pow(.5)
    an_cos = cosim(f_A, f_N)  # .pow(.5)

    # ap_D = torch.acos(ap_cos)  
    # an_D = torch.acos(an_cos)
    # print(ap_D)
    # print(an_D)

    losses = F.relu(ap_cos - an_cos) # always 0.5
    return torch.mean(losses)

def triplet_loss_fake(emb, emb_fake, y, margin, typen, size=0): # not corresponding
    with torch.no_grad():
        triplets = get_triplets(emb, y, margin, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]
    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses1 = F.relu(ap_D - an_D + margin) # always 0.5
    loss1 = torch.mean(losses1)

    if emb_fake.size(0)<f_A.size(0):
      bs=emb_fake.size(0)
    else:
      bs=f_A.size(0)

    f_a = f_A[:bs]
    f_fake = emb_fake[:bs]
    ap_f = ap_D[:emb_fake.size(0)]
    an_f = (f_a - f_fake).pow(2).sum(1)  # .pow(.5)
    losses2 = F.relu(ap_f - an_f + margin) # always 0.5
    loss2 = torch.mean(losses2)

    return (loss1+loss2)/2

def triplet_loss_allneg(emb, emb_fake, y, k, margin, typen, size=0):
    with torch.no_grad():
        triplets = get_triplets_allneg(emb, y, k, margin, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]
    if k==2:
      f_N2 = emb[triplets[:, 3]]
      ap_D = (f_A - f_P).pow(2).sum(1)  
      an_D = (f_A - f_N).pow(2).sum(1)  
      an_D2 = (f_A - f_N2).pow(2).sum(1) 
      losses1 = F.relu(ap_D - an_D + margin) 
      losses2 = F.relu(ap_D - an_D2 + margin)
      return torch.mean(losses1+losses2)/k

    elif k==3:
      f_N2 = emb[triplets[:, 3]]
      f_N3 = emb[triplets[:, 4]]  
      ap_D = (f_A - f_P).pow(2).sum(1)  
      an_D = (f_A - f_N).pow(2).sum(1)  
      an_D2 = (f_A - f_N2).pow(2).sum(1) 
      an_D3 = (f_A - f_N3).pow(2).sum(1) 
      losses1 = F.relu(ap_D - an_D + margin) 
      losses2 = F.relu(ap_D - an_D2 + margin)
      losses3 = F.relu(ap_D - an_D3 + margin)
      return torch.mean(losses1+losses2+losses3)/k

def triplet_loss_unsup(emb, emb_cutmix, bs, margin):
    fea = torch.cat([emb, emb_cutmix], dim=0)
    similarity_matrix = torch.norm(fea[:,None]-fea,dim=2,p=2)
    r_pos = torch.diag(similarity_matrix, bs)
    l_pos = torch.diag(similarity_matrix, -bs)
    positives = torch.cat([l_pos, r_pos]).view(2 * bs, 1)
    mask_samples_from_same_repr = get_correlated_mask(bs).type(torch.uint8)
    negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * bs, -1)
    positives = positives.repeat(1,negatives.size()[1])

    losses = F.relu(positives - negatives + margin) # always 0.5
    return torch.mean(losses)

def triplet_loss(emb, y, margin, typen, size=0):
    with torch.no_grad():
        triplets = get_triplets(emb, y, margin, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    # zis = F.normalize(zis, dim=1)
    # zjs = F.normalize(zjs, dim=1)
    # loss = self.nt_xent_criterion(zis, zjs)

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + margin) # always 0.5
    return torch.mean(losses)

def triplet_loss2(emb, y):
    with torch.no_grad():
        triplets = get_triplets2(y)
    if triplets.size()==torch.Size([0]):
      print("no triplet")
      return 0.

    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 0.5)
    return torch.mean(losses)

def triplet_loss3(emb, y):
    with torch.no_grad():
        triplets = get_triplets3(emb, y)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 0.5)
    return torch.mean(losses)

# pretrain
def triplet_loss_pretrain(emb_pre, emb, y, typen, size=0):
    with torch.no_grad():
        triplets = get_triplets(emb_pre, y, typen, size)
    f_A = emb[triplets[:, 0]]
    f_P = emb[triplets[:, 1]]
    f_N = emb[triplets[:, 2]]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 0.5)
    return torch.mean(losses)

def center_loss(tgt_model, batch, src_model, src_centers, tgt_centers, src_kmeans, tgt_kmeans, margin=1):
    f_N_clf = tgt_model.convnet(batch["X"].cuda()).view(batch["X"].shape[0], -1)
    f_N = tgt_model.fc(f_N_clf.detach())

    y_src = src_kmeans.predict(f_N.detach().cpu().numpy())
    ap_distances = (src_centers[y_src] - f_N).pow(2).sum(1)
    
    losses = ap_distances.mean()
  
    return losses

### Triplets Utils
def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()            
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels
    
def get_triplets(embeddings, y, margin, typen, size):
  # margin = 0.5
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel() #->1d-array
  trip = []

  for label in set(y):
    '''
    [False False False False  True  True  True False False False]
    [4 5 6]
    '''
    label_mask = (y == label)
    label_indices = np.where(label_mask)[0]

    if len(label_indices) < 2:
      ap = [label_indices.repeat(2)]
      ap = np.array(ap)
      neg_ind = np.where(np.logical_not(label_mask))[0]
      continue

    neg_ind = np.where(np.logical_not(label_mask))[0]
    ap = list(combinations(label_indices, 2))  # All anchor-positive pairs, no redundancy
    ap = np.array(ap)

    ap_D = D[ap[:, 0], ap[:, 1]]
    
    # GET HARD NEGATIVE
    # if np.random.rand() < 0.5:
    #   trip += get_neg_hard(neg_ind, hardest_negative,
    #                D, ap, ap_D, margin)
    # else:
    if typen=="h":
      trip += get_neg_hard(neg_ind, hardest_negative,
               D, ap, ap_D, margin)
    elif typen == "r":
      trip += get_neg_hard(neg_ind, random_neg,
               D, ap, ap_D, margin)
    else:
      if np.random.rand() < 0.5:
        trip += get_neg_hard(neg_ind, hardest_negative,
                     D, ap, ap_D, margin)
      else:
        trip += get_neg_hard(neg_ind, random_neg,
                     D, ap, ap_D, margin)

  if len(trip) == 0:
    ap = ap[0]
    trip.append([ap[0], ap[1], neg_ind[0]])
  elif size and len(trip) > size: # only take the first "size"
    trip = trip[:size]

  trip = np.array(trip)

  return torch.LongTensor(trip)
   
def get_triplets_allneg(embeddings, y, k, margin, typen, size):
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel() #->1d-array
  trip = []

  for label in set(y):
    label_mask = (y == label)
    label_indices = np.where(label_mask)[0]
    if len(label_indices) < 2:
        continue
    neg_ind = np.where(np.logical_not(label_mask))[0]
    
    ap = list(combinations(label_indices, 2))  # All anchor-positive pairs, no redundancy
    ap = np.array(ap)

    ap_D = D[ap[:, 0], ap[:, 1]]

    trip += get_neg_hard_allneg(neg_ind, randomk_neg,
               D, ap, ap_D, margin, k)

  if len(trip) == 0:
    ap = ap[0]
    trip.append([ap[0], ap[1], neg_ind[0], neg_ind[1]])

  trip = np.array(trip)

  return torch.LongTensor(trip)

def get_triplets2(y): #make random positive bigger
  margin = 0.5

  y = y.cpu().data.numpy().ravel() #->1d-array
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 3:
          continue
      
      ap = list(combinations(label_indices, 3))  # All anchor-positive pairs
      ap = np.array(ap)
      # print(ap)
      trip.append([ap[0][0], ap[0][1], ap[0][2]])

  if trip==[]:
    print("no triplet")
    return torch.LongTensor([])

  trip = np.array(trip)

  return torch.LongTensor(trip)
   
def get_triplets3(embeddings, y): #make small positive bigger
  margin = 0.5
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel() #->1d-array
  trip = []

  for label in set(y):
    label_mask = (y == label)
    label_indices = np.where(label_mask)[0]
    if len(label_indices) < 2:
      continue
    neg_ind = np.where(np.logical_not(label_mask))[0]
    
    ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
    ap = np.array(ap)

    ap_D = D[ap[:, 0], ap[:, 1]]
    
    trippp = []
    # ap nx2 | ap_D nx1
    ap_max = -1
    i_max = -1
    for ap_i, ap_di in zip(ap, ap_D):
      if ap_di > ap_max:
        ap_max = ap_di
        i_max = ap_i
      else:
        continue

    ap_min = ap_max
    i_min = ap_i
    if i_max.any() != -1:
      for ap_i, ap_di in zip(ap, ap_D):
        if ap_i[0] == i_max[0] and ap_di < ap_min: # same anchor
          ap_min = ap_di
          i_min = ap_i
        else:
          continue

    trippp.append([i_max[0], i_max[1], i_min[1]])
    trip += trippp

  trip = np.array(trip)
  return torch.LongTensor(trip)

def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) 
    D += vectors.pow(2).sum(dim=1).view(1, -1) 
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D

def get_triplets_cos(embeddings, y, typen, size):
  margin = 0
  D = pdist_cos(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel() #->1d-array
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
          continue
      neg_ind = np.where(np.logical_not(label_mask))[0]
      
      ap = list(combinations(label_indices, 2))  # All anchor-positive pairs, no redundancy
      ap = np.array(ap)

      ap_D = D[ap[:, 0], ap[:, 1]]
      
      if typen=="h":
        trip += get_neg_hard(neg_ind, hardest_negative,
                 D, ap, ap_D, margin)
      elif typen == "r":
        trip += get_neg_hard(neg_ind, random_neg,
                 D, ap, ap_D, margin)
      else:
        if np.random.rand() < 0.5:
          trip += get_neg_hard(neg_ind, hardest_negative,
                       D, ap, ap_D, margin)
        else:
          trip += get_neg_hard(neg_ind, random_neg,
                       D, ap, ap_D, margin)


  if len(trip) == 0:
      ap = ap[0]
      trip.append([ap[0], ap[1], neg_ind[0]])
  elif size and len(trip) > size: # only take the first "size"
      trip = trip[:size]

  trip = np.array(trip)

  return torch.LongTensor(trip)

def pdist_cos(vectors):
  bs = vectors.shape[0]
  D = torch.zeros([bs,bs])
  for i in range(bs):
    for j in range(bs):
      # torch.unsqueeze(vectors[i], 0)
      # torch.unsqueeze(vectors[i], 0)
      D[i][j] = cosim0(vectors[i], vectors[j])


  return D

def get_neg_hard(neg_ind, select_func, D, ap, ap_D, margin):
    trip = []
    # ap nx2 | ap_D nx1
    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), 
                torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def get_neg_hard_allneg(neg_ind, select_func, D, ap, ap_D, margin, k):
    trip = []
    # ap nx2 | ap_D nx1
    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values, k)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            if k == 2:
              trip.append([ap_i[0], ap_i[1], neg_hard[0], neg_hard[1]])
            elif k == 3:
              trip.append([ap_i[0], ap_i[1], neg_hard[0], neg_hard[1], neg_hard[2]])

    return trip

def topk_neg(loss_values, k):
    ind = np.argpartition(loss_values,-3)[-3:]
    return ind if loss_values[ind] > 0 else None

def randomk_neg(loss_values, k):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards, size=k, replace=True) if len(neg_hards) > 0 else None


def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def semihard_negative(loss_values, margin=1):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None