import numpy as np
from PIL import Image
import torch

def trainModel(model, device, loader, loss_list, criterion, optimizer, epoch, scheduler = None, save = None):
    """Runs a training loop for a single epoch.

    """
    # Turn on training mode
    model.train(True)

    # track loss for this epoch
    epoch_loss = []

    # loop through all batches
    for j, (img_batch, msk_batch, img_name) in enumerate(loader):
        # img needs shape [batch_size, channels, height, width]
        # mask needs [batch, H, W]

        img_batch, msk_batch = img_batch.to(device), msk_batch.to(device)

        # reset gradients
        optimizer.zero_grad()

        # process batch through network
        out = model(img_batch.float())

        # calculate loss
        loss_val = criterion(out.to(device), msk_batch.type(torch.long))  

        # Create an argmax map for a visualization of the results
        map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
        map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
        
        # track loss
        loss_list.append(loss_val.item())
        epoch_loss.append(loss_val.item())

        # backpropagation
        loss_val.backward()

        # update the parameters
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if save:
            map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
            # Create an argmax map
            #map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            #plt.imshow(map_out)
            #map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)
            im = Image.fromarray(map_out)
            im.save(save % (epoch, img_name[0]))

    print("*" * 20)
    print('Training loss for epoch %d : ' % epoch, np.mean(np.array(epoch_loss)))
    print()

    return loss_list

def validateModel(model, device, loader, loss_list, epoch, criterion, save = None):
    """Runs a validation loop for a single epoch.

    """
    # Turn off training mode
    model.train(False)

    # track loss for this epoch
    epoch_loss = []

    # loop through all batches
    with torch.no_grad():
        
        for img_batch, msk_batch, img_name in loader:

            # load image and mask
            img_batch, msk_batch = img_batch.to(device), msk_batch.to(device)

            # process batch through network
            out = model(img_batch.float())

            # calculate loss
            loss_val = criterion(out.to(device), msk_batch.type(torch.long))  
            
            # track loss
            loss_list.append(loss_val.item())
            epoch_loss.append(loss_val.item())

            map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
            map_out = np.where(map[0,:,:]>map[1,:,:], 0, 255).astype(np.uint8)

            if save:
                # Create an argmax map
                #map = np.squeeze(out.cpu().detach().numpy()[0,:,:,:])
                im = Image.fromarray(map_out)
                im.save(save % (epoch, img_name[0]))

        print("*" * 20)
	print('Validation loss for epoch %d : ' % epoch, np.mean(np.array(epoch_loss)))
        print()

        return loss_list
