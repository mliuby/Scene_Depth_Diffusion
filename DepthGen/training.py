import time
import datetime
import os
import torch
import torchvision.utils as vision_utils
from tensorboardX import SummaryWriter
from ddpm import DDPM
from tools import AverageMeter, load_from_checkpoint, init_or_load_model,interpolation


model_prefix = "DepthGen_"
def train(epochs,
        trainloader,
        testloader,
        lr=3e-5,
        batch_size=1,
        betas=(1e-4, 0.02),
        n_T=128,
        save="/content/drive/MyDrive/checkpoints",#"/checkpoints"
        device="cuda",
        checkpoint=None,
        criterion=torch.nn.L1Loss(),
        need_interpolation=True,
        unroll_step=True):

    device = torch.device("cuda:0" if device == "cuda" else "cpu")
    num_trainloader = len(trainloader)
    num_testloader = len(testloader)

    if not os.path.exists('/content/drive/MyDrive/checkpoints'):##"/checkpoints"
        os.makedirs('/content/drive/MyDrive/checkpoints')   
    if checkpoint:
        print("Loading from checkpoint ...")
        model, optim, start_epoch = init_or_load_model(epochs=epochs,lr=lr,ckpt=checkpoint,device=device)
        print("Resuming from: epoch #{}".format(start_epoch))
    else:
        print("Initializing fresh model ...")
        model, optim, start_epoch = init_or_load_model(epochs=epochs,lr=lr,ckpt=None,device=device)
    

    log_dir = '/content/drive/MyDrive/runs/not_pretrained'
    writer = SummaryWriter(log_dir,comment="{}training".format(model_prefix))
    print("Starting training ... ")

    for epoch in range(start_epoch,epochs):
        ddpm=DDPM(model, (1e-4, 0.02), n_T, device, criterion)
        ddpm.to(device)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        epoch_start = time.time()
        end = time.time()
        # linear lrate decay
        #optim.param_groups[0]['lr'] = lr*(1-ep/n_epoch)
        for idx, batch in enumerate(trainloader):
            model.train()
            optim.zero_grad()
            image_x = batch["image"].to(device)
            depth_y = batch["depth"].to(device)
            loss= ddpm(image_x,depth_y,need_interpolation=True,unroll_step=True)
            loss_meter.update(loss.detach().item(), image_x.size(0))
            loss.backward()
            optim.step()
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(num_trainloader-idx))))
            num_iters = epoch * num_trainloader + idx
            if idx %1 == 0 :
                print(
                    "Epoch: #{0} Batch: {1}/{2}\t"
                    "Time (current/total) {batch_time.val:.3f}/{batch_time.sum:.3f}\t"
                    "eta {eta}\t"
                    "LOSS (current/average) {loss.val:.4f}/{loss.avg:.4f}\t"
                    .format(epoch, idx, num_trainloader, batch_time=batch_time, eta=eta, loss=loss_meter)
                )

                writer.add_scalar("Train/Loss", loss_meter.val, num_iters)
            if idx%200 == 0:
                ckpt_path = save+"ckpt_{}_not_pretrained.pth".format(epoch)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict":  optim.state_dict(),
                    "loss": loss_meter.avg
                }, ckpt_path)
            if idx%400==0: LogProgress(model, writer, testloader, num_iters,n_T, device)
            del image_x
            del depth_y


        print(
            "----------------------------------\n"
            "Epoch: #{0}, Avg. Net Loss: {avg_loss:.4f}\n"
            "----------------------------------"
            .format(
                epoch, avg_loss=loss_meter.avg
            )
        )
def LogProgress(model, writer, test_loader, num_iters, n_T, device):

    model.eval()
    ddpm=DDPM(model, (1e-4, 0.02), n_T, device, criterion=torch.nn.L1Loss())
    ddpm.to(device)
    sequential = test_loader
    sample_batched = next(iter(sequential))

    image = sample_batched["image"].to(device)
    depth = sample_batched["depth"].to(device)

    if num_iters == 0:
      writer.add_image("Train.1.Image",vision_utils.make_grid(image.data,nrow=6,normalize=True),num_iters)
    if num_iters == 0:
      writer.add_image("Train.2.Image",vision_utils.make_grid(depth.data,nrow=6,normalize=False),num_iters)

    output=ddpm.sample(image,device)
    writer.add_image("Train.3.Ours",vision_utils.make_grid(output.data,nrow=6,normalize=False),num_iters)
    writer.add_image("Train.4.Diff",vision_utils.make_grid(torch.abs(output-depth).data,nrow=6,normalize=False),num_iters)


    del image
    del depth
    del output
'''

from importlib import reload
import torch
path="/content/drive/MyDrive/data/nyu_rawdepth.zip" 
trainloader,testloader=get_data_loaders(path, batch_size=1)
train(     4,
        trainloader,
        testloader,
        lr=1e-5,
        batch_size=1,
        betas=(1e-4, 0.02),
        n_T=128,
        save="/content/drive/MyDrive/checkpoints",
        device="cuda",
        checkpoint=None,
        criterion=torch.nn.L1Loss())
'''        
'''
%load_ext tensorboard
%tensorboard --logdir '/content/drive/MyDrive/runs/not_pretrained'
'''



