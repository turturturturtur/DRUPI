import torch
import torch.nn as nn
import torch.nn.functional as F

def approx_infoNCE_loss(q, k,label):
    # 计算 query 和 key 的相似度得分
    similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分

    # 计算相似度得分的温度参数
    temperature = 0.07

    # 计算 logits
    logits = similarity_scores / temperature

    # # 构建 labels（假设有N个样本）
    # N = q.size(0)
    # labels = torch.arange(N).to(logits.device)

    # 计算交叉熵损失
    loss = F.cross_entropy(logits, label)

    return loss

def compute_info_nce_loss(feat_syn, label_syn, c, num_classes=10, samples_per_class=10, temperature=0.07):
    """
    Compute the InfoNCE loss for contrastive learning.

    Args:
        feat_syn: Tensor of shape (total_samples, channels, height, width) containing feature vectors.
        label_syn: Tensor of shape (total_samples,) containing labels for each feature vector.
        c: The class label considered as the positive sample.
        num_classes: The total number of classes.
        samples_per_class: The number of samples per class.
        temperature: Scaling factor for the similarity scores.

    Returns:
        loss: The InfoNCE loss.
    """
    total_samples = num_classes * samples_per_class

    # 获取类别为 c 的样本特征和负样本特征
    pos_indices = torch.arange(c * samples_per_class, (c + 1) * samples_per_class)
    neg_indices = torch.cat([torch.arange(0, c * samples_per_class), torch.arange((c + 1) * samples_per_class, total_samples)])

    feat_syn_pos = feat_syn[pos_indices]
    feat_syn_neg = feat_syn[neg_indices]

    # 调整特征张量形状
    feat_syn_pos = feat_syn_pos.view(feat_syn_pos.size(0), -1)
    feat_syn_neg = feat_syn_neg.view(feat_syn_neg.size(0), -1)
    
    total_loss = 0
    num_batches = 0
    
    for i in range(0, feat_syn_neg.size(0), 10):
        if i + 10 <= feat_syn_neg.size(0):
            k_batch = feat_syn_neg[i:i+10]
            loss = approx_infoNCE_loss(feat_syn_pos, k_batch,label_syn[pos_indices])
            total_loss += loss.item()
            num_batches += 1
    
    average_loss = total_loss / num_batches
    return average_loss

    
def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    # check the shape of the sample, convert it into (bs, d)
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1
    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


class MMDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return MMD(input, target, kernel="rbf")


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.view(input.size(0), -1)
        target = target.view(target.size(0), -1)
        input = input / input.norm(dim=1)[:, None]
        target = target / target.norm(dim=1)[:, None]
        return 1 - torch.mean(torch.sum(input * target, dim=1))




import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode#设置对比的模式有one和all两种，代表对比一个channel还是所有，个人理解
        self.base_temperature = base_temperature #设置的温度

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')#设置设备
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:# batch_size, channel,H,W，平铺变成batch_size, channel, (H,W)
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:#只能存在一个
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:#如果两个都没有就是无监督对比损失，mask就是一个单位阵
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:#有标签，就把他变成mask
            labels = labels.contiguous().view(-1, 1)#contiguous深拷贝，与原来的labels没有关系，展开成一列,这样的话能够计算mask，否则labels一维的话labels.T是他本身捕获发生转置
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask =  torch.eq(labels, labels.T).float().to(device)#label和label的转置比较，感觉应该是广播机制，让label和label.T都扩充了然后进行比较，相同的是1，不同是0.
            #这里就是由label形成mask,mask(i,j)代表第i个数据和第j个数据的关系，如果两个类别相同就是1， 不同就是0
        else:
            mask = mask.float().to(device)#有mask就直接用mask，mask也是代表两个数据之间的关系

        contrast_count = features.shape[1]#对比数是channel的个数
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)#把feature按照第1维拆开，然后在第0维上cat，(batch_size*channel,h*w..)#后面就是展开的feature的维度
        #这个操作就和后面mask.repeat对上了，这个操作是第一个数据的第一维特征+第二个数据的第一维特征+第三个数据的第一维特征这样排列的与mask对应
        if self.contrast_mode == 'one':#如果mode=one，比较feature中第1维中的0号元素(batch, h*w)
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':#all就(batch*channel, h*w)
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),#两个相乘获得相似度矩阵，乘积值越大代表越相关
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)#计算其中最大值
        logits = anchor_dot_contrast - logits_max.detach()#减去最大值，都是负的了，指数就小于等于1

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)#repeat它就是把mask复制很多份
        # mask-out self-contrast cases
        logits_mask = torch.scatter(#生成一个mask形状的矩阵除了对角线上的元素是0，其他位置都是1， 不会对自身进行比较
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask#定义其中的相似度
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))#softmax

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)#mask的和
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, torch.tensor(1, dtype=mask_pos_pairs.dtype,device=mask_pos_pairs.device), mask_pos_pairs)#满足返回1，不满足返回mask_pos_pairs.保证数值稳定
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos#类似蒸馏temperature温度越高，分布曲线越平滑不易陷入局部最优解，温度低，分布陡峭
        loss = loss.view(anchor_count, batch_size).mean()#计算平均

        return loss

# Example usage:
# sup_con_loss = SupConLoss(temperature=0.07, contrast_mode='all')
# loss = sup_con_loss(feat_syn, label_syn, c)


def likelihood(mu, log_var, tgt_embed, eps=1e-5):
    return ((mu - tgt_embed) ** 2 / (log_var.exp() + eps) - log_var).mean()


def club(mu, log_var, tgt_embed, eps=1e-5):
    random_idx = torch.randperm(mu.size(0)).long().to(mu.device)
    positive = - (mu - tgt_embed) ** 2 / (log_var.exp() + eps)
    negative = - (mu - tgt_embed[random_idx]) ** 2 / (log_var.exp() + eps)
    return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean() / 2.
