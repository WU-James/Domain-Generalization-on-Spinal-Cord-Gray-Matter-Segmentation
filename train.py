import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import utils
import models
import metrics


def do_train(train_loader, val_loader, device, load_model, model_path, test_domain, epochs, lr, momentum,
          fourier_transform, train_dataset, encoder, log):
    # initialize model and max dice
    model = models.get_model(device, encoder)
    if fourier_transform:
        model_path = model_path + test_domain + "_as_test_fourier.pt"
    else:
        model_path = model_path + test_domain + "_as_test.pt"
    max_dice = 0.0

    # if needed, load model and max dice
    if load_model:
        model.load_state_dict(torch.load(model_path)["params"])
        max_dice = torch.load(model_path)["dice"]["avg"]
        print(("Model loaded. Max dice: {:.4f}. Loaded path: " + model_path).format(max_dice))

    # loss and optimizer
    criterion = metrics.DiceLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # epoch
    for epoch in range(epochs):
        print(" -- Epoch {}/{}".format(epoch + 1, epochs))

        # training
        model.train()

        running_loss_train = 0.0
        prediction_list_trian = []
        spinal_cord_mask_list_trian = []
        gm_mask_list_trian = []

        # batch loop
        for data in tqdm(train_loader):
            # forward and backward
            optimizer.zero_grad()
            images, spinal_cord_masks, gm_masks, domains = data
            if fourier_transform:
                images, spinal_cord_masks, gm_masks, domains = utils.get_augmented(images, spinal_cord_masks, gm_masks,
                                                                                   domains,
                                                                                   train_dataset)  # N*C*H*W -> 4N*C*H*W

            images = images.to(device)
            spinal_cord_masks = spinal_cord_masks.to(device)
            gm_masks = gm_masks.to(device)
            outputs = model(images)
            outputs = F.sigmoid(outputs)

            dice_loss1 = criterion(outputs[:, 0, :, :], spinal_cord_masks)
            dice_loss2 = criterion(outputs[:, 1, :, :], gm_masks)
            loss = (dice_loss1 + dice_loss2) / 2.0

            loss.backward()
            optimizer.step()

            # update running loss and output & mask list
            running_loss_train += loss.item()

            for i in range(outputs.size(0)):
                prediction = outputs[i].detach().cpu().numpy()
                prediction = np.where(prediction >= 0.5, 1, 0)
                spinal_cord_mask = spinal_cord_masks[i].detach().cpu().numpy()
                gm_mask = gm_masks[i].detach().cpu().numpy()

                prediction_list_trian.append(prediction)
                spinal_cord_mask_list_trian.append(spinal_cord_mask)
                gm_mask_list_trian.append(gm_mask)

        # epoch loss and dice score
        loss = running_loss_train / len(train_loader)
        dice_score = metrics.compute_dice(prediction_list_trian, spinal_cord_mask_list_trian, gm_mask_list_trian)

        log["train"]["loss"].append(loss)
        log["train"]["dice_score"].append(dice_score)

        print("Training: Loss {:.4f}, Spinal Cord Dice {:.4f}, GM Dice {:.4f}".format(
            loss,
            dice_score[0],
            dice_score[1]
        ))

        # validation
        model.eval()
        prediction_list_val = []
        spinal_cord_mask_list_val = []
        gm_mask_list_val = []

        for data in val_loader:
            # forward
            images, spinal_cord_masks, gm_masks, domains = data
            images = images.to(device)
            spinal_cord_masks = spinal_cord_masks.to(device)
            gm_masks = gm_masks.to(device)
            outputs = model(images)
            outputs = F.sigmoid(outputs)

            # update and output & mask list
            for i in range(outputs.size(0)):
                prediction = outputs[i].detach().cpu().numpy()
                prediction = np.where(prediction >= 0.5, 1, 0)
                spinal_cord_mask = spinal_cord_masks[i].detach().cpu().numpy()
                gm_mask = gm_masks[i].detach().cpu().numpy()

                prediction_list_val.append(prediction)
                spinal_cord_mask_list_val.append(spinal_cord_mask)
                gm_mask_list_val.append(gm_mask)

        # epoch loss and dice score
        dice_score = metrics.compute_dice(prediction_list_val, spinal_cord_mask_list_val, gm_mask_list_val)
        # jaccard_score = compute_jaccard(prediction_list_val, spinal_cord_mask_list_val, gm_mask_list_val)
        asd_score = metrics.compute_asd(prediction_list_val, spinal_cord_mask_list_val, gm_mask_list_val)
        # hd_score = compute_hd(prediction_list_val, spinal_cord_mask_list_val, gm_mask_list_val)

        log["val"]["dice_score"].append(dice_score)
        log["val"]["asd"].append(asd_score)
        # log["val"]["jaccard"].append(jaccard_score)
        # log["val"]["hd"].append(hd_score)

        print("Validation: Spinal Cord Dice {:.4f}, GM Dice {:.4f}, Spinal Cord ASD, GM ASD {:.4f}, {:.4f}".format(
            dice_score[0], dice_score[1],
            asd_score[0], asd_score[1]
            # jaccard_score[0], jaccard_score[1],
            # hd_score[0], hd_score[1]
        ))

        # Save best model
        avg_dice = (dice_score[0] + dice_score[1]) / 2.0
        avg_asd = (asd_score[0] + asd_score[1]) / 2.0

        if avg_dice > 0.7 and avg_dice > max_dice:
            max_dice = avg_dice
            torch.save({"params": model.state_dict(),
                        "dice": {"avg": avg_dice, "data": dice_score},
                        "asd": {"avg": avg_dice, "data": asd_score}}, model_path)
            print(("**Model saved. Max dice: {:.4f}. Save path: " + model_path).format(max_dice))

    return log