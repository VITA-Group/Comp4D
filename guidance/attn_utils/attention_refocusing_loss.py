import math
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import numbers
from torch import nn


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


class AverageSmoothing(nn.Module):
    """
    Apply average smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the average kernel.
        sigma (float, sequence): Standard deviation of the rage kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, dim=2):
        super(AverageSmoothing, self).__init__()

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = torch.ones(size=(kernel_size, kernel_size)) / (kernel_size * kernel_size)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply average filter to input.
        Arguments:
            input (torch.Tensor): Input to apply average filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


def loss_one_att_outside(attn_map, bboxes):
    loss = 0
    object_number = len(bboxes)
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    # print('in loss_one_att_outside ',b, i, j, H)
    for obj_idx in range(object_number):
        
        for obj_box in bboxes[obj_idx]:
            mask = torch.zeros(size=(H, W)).cuda() if torch.cuda.is_available() else torch.zeros(size=(H, W))
            x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
            mask[y_min: y_max, x_min: x_max] = 1.
            mask_out = 1. - mask
            index = (mask == 1.).nonzero(as_tuple=False)
            index_in_key = index[:,0]* H + index[:, 1]
            att_box = torch.zeros_like(attn_map)
            att_box[:,index_in_key,:] = attn_map[:,index_in_key,:]

            att_box = att_box.sum(axis=1) / index_in_key.shape[0]
            att_box = att_box.reshape(-1, H, H)
            activation_value = (att_box* mask_out).reshape(b, -1).sum(dim=-1) #/ att_box.reshape(b, -1).sum(dim=-1)
            loss += torch.mean(activation_value)
            
    return loss / object_number

def caculate_loss_self_att(self_attention_lists, bboxes):
    cnt = 0
    total_loss = 0

    for self_att in self_attention_lists:
        if not self_att:
            continue
        for attn in self_att:
            attn = attn.view(attn.shape[0]*attn.shape[1], *attn.shape[2:])
            # print('in caculate_loss_self_att attn: ', attn.shape)
            total_loss += loss_one_att_outside(attn, bboxes)
            cnt += 1

    return total_loss /cnt

def caculate_loss_att_fixed_cnt(cross_attn_lists, bboxes, object_positions, t, res=16, smooth_att = True, sigma=0.5, kernel_size=3 ):
    result  = []
    
    for cross_attn_list in cross_attn_lists:
        if cross_attn_list == []: continue
        for attn_map in cross_attn_list:
            # print('in caculate_loss_att_fixed_cnt attn_map: ', attn_map.shape)
            H = W = int(math.sqrt(attn_map.shape[-2]))
            s = attn_map.shape[0] * attn_map.shape[1]
            attn_map = attn_map.view(attn_map.shape[0] * attn_map.shape[1], H, W, attn_map.shape[-1]).sum(0) / s
            # print('in caculate_loss_att_fixed_cnt attn_map view: ', attn_map.shape)
            result.append(attn_map)

    obj_number = len(bboxes)
    total_loss = 0
    
    for attn in result:
        attn_text = attn[:, :, 1:-1]
        attn_text *= 100
        attn_text = torch.nn.functional.softmax(attn_text, dim=-1)
        current_res = attn.shape[0]
        H = W = current_res
    
    
        min_all_inside = 1000
        max_outside = 0
        
        for obj_idx in range(obj_number):
        
            for obj_position in object_positions[obj_idx]:
                true_obj_position = obj_position - 1
                att_map_obj = attn_text[:,:, true_obj_position]
                if smooth_att:
                    smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                    input = F.pad(att_map_obj.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                    att_map_obj = smoothing(input).squeeze(0).squeeze(0)
                other_att_map_obj = att_map_obj.clone()
                att_copy = att_map_obj.clone()

                for obj_box in bboxes[obj_idx]:
                    x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                
                
                    if att_map_obj[y_min: y_max, x_min: x_max].numel() == 0: 
                        max_inside=1.
                        
                    else:
                        max_inside = att_map_obj[y_min: y_max, x_min: x_max].max()
                    if max_inside < 0.1:
                        total_loss += 6*(1. - max_inside)
                    elif max_inside < 0.2:
                        total_loss += 1. - max_inside
                    elif t < 15:
                        total_loss += 1. - max_inside
                    if max_inside < min_all_inside:
                        min_all_inside = max_inside
                    
                    # find max outside the box, find in the other boxes
                    
                    att_copy[y_min: y_max, x_min: x_max] = 0.
                    other_att_map_obj[y_min: y_max, x_min: x_max] = 0.
                
                for obj_outside in range(obj_number):
                    if obj_outside != obj_idx:
                        for obj_out_box in bboxes[obj_outside]:
                            x_min_out, y_min_out, x_max_out, y_max_out = int(obj_out_box[0] * W), \
                                int(obj_out_box[1] * H), int(obj_out_box[2] * W), int(obj_out_box[3] * H)
                            
                    
                            if other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].numel() == 0: 
                                max_outside_one= 0
                            else:
                                max_outside_one = other_att_map_obj[y_min_out: y_max_out, x_min_out: x_max_out].max()
                            
                            att_copy[y_min_out: y_max_out, x_min_out: x_max_out] = 0.
                            
                            if max_outside_one > 0.15:
                                total_loss += 4 * max_outside_one
                            elif max_outside_one > 0.1:
                                total_loss += max_outside_one
                            elif t<15:
                                total_loss += max_outside_one
                            if max_outside_one > max_outside:
                                max_outside = max_outside_one
                
                max_background = att_copy.max()
                total_loss += len(bboxes[obj_idx]) * max_background /2.
                
    return total_loss/obj_number, min_all_inside, max_outside


