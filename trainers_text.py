import torch
import torch.nn as nn
import time
from utils import AverageMeter, ProgressMeter, accuracy
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from torch.autograd import Variable
def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)



def ssl_with_text(
    textmodel,
    model,
    device,
    dataloader,
    criterion,
    criterion_MSE,
    criterion_CL,
    optimizer,
    optimizer_text,
    optimizer_cross,
    lr_scheduler=None,
    lr_scheduler_cross=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    textmodel.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        caps = data#.to(device)


        # print(images[0].shape)
        # print(images[1].shape)
        cap_single = caps[0].to(device)
        caps = torch.cat([caps[0], caps[1]], dim=0).to(device)
        # caps = torch.cat([caps[0], caps[1]], dim=0).to(device)
        caps = caps.to(device)
        bsz = caps.shape[0]//2
        # basic properties of training
        if i == 0:
            print(
                caps.shape,
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )
            print(
                "Pixel range for training images : [{}, {}]".format(
                    torch.min(caps).data.cpu().numpy(),
                    torch.max(caps).data.cpu().numpy(),
                )
            )

        features = textmodel(caps)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features)
        losses.update(loss.item(), bsz)
        # print("1",loss)


        # single_features = torch.nn.functional.normalize(single_features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # features_cross = torch.cat([If1.unsqueeze(1), cap_features.unsqueeze(1)], dim=1)

        optimizer_text.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_text.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



        # textmodel.train()
        # cap_features = textmodel(caps)
        # # f1_cap, f2_cap = torch.split(cap_features, [bsz, bsz], dim=0)

        # If1 = torch.nn.functional.normalize(f1.clone().detach(),dim=1)
        # If2 = torch.nn.functional.normalize(f2.clone().detach(),dim=1)
 
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # # cap_features = torch.cat([f1_cap.unsqueeze(1), f2_cap.unsqueeze(1)], dim=1)

        # loss_cross_1 = criterion_MSE(If1,cap_features)
        # loss_cross_2 = criterion_MSE(If2,cap_features)
        # loss_cross = loss_cross_1 + loss_cross_2
        # optimizer_text.zero_grad()
        # loss_cross.backward(retain_graph=True)
        # optimizer_text.step()
        # if lr_scheduler:
        #     lr_scheduler.step()

        

        ###generate samples ###
        ###generate samples ###
        ###generate samples ###


        # model.eval(),textmodel.eval()
        # features = model(image_single)
        # cap_features = textmodel(caps)
        # # print(cap_features.shape)
        # # print(features.shape)
        # features = torch.nn.functional.normalize(features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # # print(features.shape)
        # # print(cap_features.shape)
        # cor_matrix = torch.mm( cap_features, features.t() )
        # cor_matrix = cor_matrix / torch.sum(cor_matrix,dim=1)
        # generated_batch = []
        # for cap_i in cor_matrix:
        #     generate_img = torch.zeros(images[0].shape).to(device)
        #     for imgid, cap_w in enumerate(cap_i):
        #         generate_img += cap_w * image_single[imgid]
        #     generated_batch.append(generate_img)
        # # print(len(generated_batch))

        # model.train(),textmodel.eval()
        # # generated_batch = torch.tensor([item.cpu().detach().numpy() for item in generated_batch]).cuda() 

        # images_ = Generated_image(generated_batch, bsz, size = args.size )
        # for images in images_:
        #     images = torch.cat([images[0], images[1]], dim=0).to(device)
            


        #     features = model(images)
        #     f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        #     features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
        #     loss = criterion(features)
        #     losses.update(loss.item(), bsz)

        #     # print(loss)
        #     optimizer.zero_grad()
        #     loss.backward(retain_graph=True)
        #     optimizer.step()




def ssl_with_text_CMA_CRA(
    textmodel,
    model,
    device,
    dataloader,
    criterion,
    criterion_MSE,
    criterion_CL,
    optimizer,
    optimizer_text,
    optimizer_cross,
    lr_scheduler=None,
    lr_scheduler_cross=None,
    epoch=0,
    args=None,
):
    print(
        " ->->->->->->->->->-> One epoch with self-supervised training <-<-<-<-<-<-<-<-<-<-"
    )

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    textmodel.train()
    end = time.time()

    for i, data in enumerate(dataloader):
        caps, images = data#.to(device)


        # gmm = GMM(n_components=5).fit(caps[0])
        # center = gmm.means_
        # # label = gmm.predict_proba(caps)
        # label = gmm.predict(caps[0])
        # l = label.shape[0]
        # label = np.array(list(label) * l ).reshape(l,l)
        # mask = torch.FloatTensor( (label == label.T) * 1 )
        

        caps = torch.cat([caps[0], caps[1]], dim=0).to(device)
        # caps = torch.cat([caps[0], caps[1]], dim=0).to(device)
        caps = caps.to(device)
        bsz = caps.shape[0]//2
 
        images = images.to(device)
        caps_single = caps[:bsz]

        features = textmodel(caps)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features)
        losses.update(loss.item(), bsz)


        img_features = model(images)

        features_cross = torch.cat([f1.unsqueeze(1), img_features.unsqueeze(1)], dim=1)
        loss_cross = criterion(features_cross)
        # print("A:",loss)
        # print("B:",loss_cross)
        loss = loss + 0.1 * loss_cross
        optimizer_cross.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_cross.step()
        # if lr_scheduler:
        #     lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        

        ###generate samples ###
        ###generate samples ###
        ###generate samples ###

        # model.eval(),textmodel.eval()
        with torch.no_grad():
            features = textmodel(caps_single)
            img_features = model(images)
        # print(cap_features.shape)
        # print(features.shape)
        # features = torch.nn.functional.normalize(features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # print(features.shape)
        # print(cap_features.shape)
            z_i = F.normalize(features, dim=1)
            z_j = F.normalize(img_features, dim=1)
        # representations = torch.cat([z_i, z_j], dim=0)
        # cor_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)
        # cor_matrix = F.cosine_similarity(z_i, z_j, dim=1)
            cor_matrix = torch.abs(torch.mm( z_j, z_i.t() ))
            # cor_matrix = cor_matrix - torch.diag( torch.diag(cor_matrix) ) + 10 * torch.eye(cor_matrix.shape[0]).to(device)
            # cor_matrix = torch.mul(mask.to(device), cor_matrix)

            # cor_matrix = nn.Softmax(cor_matrix,dim=1)
            cor_matrix = cor_matrix / torch.sum(cor_matrix,dim=1)
            # print(cor_matrix)
            generated_batch = []
            for cap_i in cor_matrix:
                generate_cap = torch.zeros(caps_single[0].shape).to(device)
                for capid, cap_w in enumerate(cap_i):
                    # if cap_w == 0.0:
                    #     continue

                    generate_cap += cap_w * caps_single[capid]
                    
                generated_batch.append(generate_cap)


        textmodel.train()
        generated_batch = torch.tensor([item.cpu().detach().numpy() for item in generated_batch]).cuda() 
        generated_I_features = textmodel(generated_batch)
        # cap_features = textmodel(caps)
        # generated_I_features = torch.nn.functional.normalize(generated_I_features,dim=1)
        # cap_features = torch.nn.functional.normalize(cap_features,dim=1)
        f2 = textmodel(caps[bsz:])
        features = torch.cat([generated_I_features.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        
        loss = criterion(features)
        # print("1",loss)

        # print(loss)

        optimizer_text.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_text.step()

        if lr_scheduler:
            lr_scheduler.step()
            lr_scheduler_cross.step()
