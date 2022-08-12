
import torch 
import torch.nn as nn 
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, n_class, epsilon=1e-7):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.n_class = n_class

    def forward(self, input, target):
        """
        :param input: logits: N, C, H, W
        :param target: labels: N, H, W
        """
        IGNORE_INDEX = -1
        mask = target != IGNORE_INDEX
        mask = mask.unsqueeze(1).expand(-1, self.n_class, -1, -1).float()

        input = F.softmax(input, dim=1)
        input = input * mask
        dim = tuple(range(1, len(input.shape) - 1))

        tt = self._make_one_hot(target).float()

        numerator = 2.0 * torch.sum(input * tt, dim=(2, 3))
        denominator = torch.sum(input + tt, dim=(2, 3))
        dice = numerator / (denominator + self.epsilon)
        return 1.0 - torch.mean(dice)

    def _make_one_hot(self, labels):
        """
        labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        """
        one_hot = torch.zeros(
            (labels.shape[0], self.n_class + 1, labels.shape[1], labels.shape[2]),
            dtype=torch.float32,
        ).cuda()

        target = one_hot.scatter_(1, (labels + 1).unsqueeze(1).data, 1)
        target = target[:, 1:, :, :]
        return target

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        self.size_average = size_average

    def forward(self, input, target, ignore_index=-1):
        
        """
        :param input: logits: N, C, H, W
        :param target: labels: N, H, W 
        :param ignore_index: int (class label to ignore)
        :return:
        """

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        log_p_t = F.log_softmax(input, dim=-1)

        target = target.view(-1, 1)  # N,H,W => N*H*W

        # ignore
        mask = target == ignore_index
        mask = mask.view(-1)
        target[mask] = 0 
        log_p_t = log_p_t.gather(1, target)  # regular Cross-Entropy
        log_p_t = log_p_t.view(-1)
        log_p_t[mask] = 0  # (+++) this line makes ignore_class work

        p_t = F.softmax(input, dim=-1)
        p_t = p_t.gather(1, target)
        p_t = p_t.view(-1)

        loss = -1 * (1 - p_t).pow(self.gamma) * log_p_t

        if self.alpha is not None:
            target = target.view(-1).long()
            alpha_t = self.alpha.gather(0, target)  # ignored are already 0 (+++)
            loss = loss * alpha_t

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

if __name__ == "__main__":
    #unit test 

    input = torch.randn(1, 2, 3, 3)
    target = torch.randint(0, 2, (1, 3, 3))
    loss = FocalLoss()
    loss1 = DiceLoss(n_class =2)

    print(loss(input, target))
    print(loss1(input, target))