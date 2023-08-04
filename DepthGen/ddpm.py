
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from Unet import Unet
from tools import  AverageMeter, load_from_checkpoint, init_or_load_model,interpolation





def ddpm_schedules(beta1, beta2, T):

    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta = torch.sqrt(beta)
    alpha = 1 - beta
    log_alpha = torch.log(alpha)
    alphabar = torch.cumsum(log_alpha, dim=0).exp()

    sqrt_alphabar = torch.sqrt(alphabar)
    one_over_sqrt_alpha= 1 / torch.sqrt(alpha)

    sqrt_one_minus_alphabar = torch.sqrt(1 - alphabar)
    one_minus_alpha_over_sqrt_one_minus_alphabar = (1 - alpha) / sqrt_one_minus_alphabar
    one_over_sqrt_alphabar=1/sqrt_alphabar
    sqrt_one_minus_alpha=torch.sqrt(1-alpha)
    return {
        "alpha": alpha, 
        "one_over_sqrt_alpha": one_over_sqrt_alpha, 
        "sqrt_beta": sqrt_beta, 
        "alphabar": alphabar,  
        "sqrt_alphabar": sqrt_alphabar, 
        "sqrt_one_minus_alphabar": sqrt_one_minus_alphabar,  
        "one_minus_alpha_over_sqrt_one_minus_alphabar": one_minus_alpha_over_sqrt_one_minus_alphabar,  
        "one_over_sqrt_alphabar":one_over_sqrt_alphabar,
        "sqrt_one_minus_alpha":sqrt_one_minus_alpha
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device,criterion):
        super().__init__()
        self.nn_model = nn_model.to(device)
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.device = device
        self.criterion = criterion
    def forward(self, image_x, depth_y, need_interpolation=False, unroll_step=False):
        if need_interpolation:
            mask = (depth_y!= 0)  # missing depth represented by 0
            mask =torch.tensor(mask).to(self.device)
            depth_y=interpolation(depth_y.to(self.device),delta_h=20)

        depth_y=depth_y*2-1  #rescale depth_y，image_x  from (0,1)to (-1,1)
        image_x=image_x*2-1 
        _ts = torch.randint(1, self.n_T+1, (depth_y.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(depth_y)  # eps ~ N(0, 1)
        depth_y_t = (self.sqrt_alphabar[_ts] * depth_y + self.sqrt_one_minus_alphabar[_ts] * noise)
        if unroll_step:
            with torch.no_grad():
                image_x_depth_y_t=torch.cat((image_x,depth_y_t),dim=1)
                pred_noise=self.nn_model(image_x_depth_y_t ,_ts / self.n_T)
            pred_depth_y=(depth_y_t-self.sqrt_one_minus_alphabar[_ts]*pred_noise)*self.one_over_sqrt_alphabar[_ts]
            depth_y_t=self.sqrt_alphabar[_ts]*pred_depth_y+self.sqrt_one_minus_alphabar[_ts]*noise
            noise=(depth_y_t-depth_y*self.sqrt_alphabar[_ts])/(self.sqrt_one_minus_alphabar[_ts])

        image_x_depth_y_t=torch.cat((image_x,depth_y_t),dim=1)
        pred_noise=self.nn_model(image_x_depth_y_t, _ts / self.n_T)

        if need_interpolation:
            noise=noise*mask #mask the missing depth
            pred_noise=pred_noise*mask
        
        return self.criterion(noise, pred_noise)

    def sample(self,image_x,device):
        
        image_x=image_x*2-1
        pred_depth = torch.randn_like(image_x[:, :1, :, :])
        with torch.no_grad():
            for t in range(self.n_T, 0, -1):
                print(f'sampling timestep {t}',end='\r')
                depth_image=torch.cat((image_x,pred_depth),dim=1)
                t_is = torch.tensor([t / self.n_T]).to(device)
                t_is = t_is.repeat(image_x.shape[0])
                pred_noise=self.nn_model(depth_image,  t_is).to(device)
                z =torch.randn_like(pred_noise).to(device) if t > 1 else 0
                pred_depth=(pred_depth-self.one_minus_alpha_over_sqrt_one_minus_alphabar[t]*pred_noise)\
                *self.one_over_sqrt_alpha[t]+self.sqrt_one_minus_alpha[t]*z
        pred_depth=pred_depth*0.5+0.5        
        return torch.clamp(pred_depth,0,1)
