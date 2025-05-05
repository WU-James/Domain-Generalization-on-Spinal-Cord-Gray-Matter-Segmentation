import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import utils
import models
import metrics
import segmentation_models_pytorch as smp
import pandas as pd
import dataset
from torch.utils.data import DataLoader


def do_test(device, test_domain, test_loader, model_path, encoder, fourier_transform):
    best_model = smp.Unet(
        encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your dataset)
    ).to(device)

    if (fourier_transform):
        model_path = model_path + test_domain + "_as_test_fourier.pt"
    else:
        model_path = model_path + test_domain + "_as_test.pt"

    best_model.load_state_dict(torch.load(model_path)["params"])
    print("model loaded at: " + model_path)

    best_model.eval()
    prediction_list_test = []
    spinal_cord_mask_list_test = []
    gm_mask_list_test = []

    for data in test_loader:
        # forward
        images, spinal_cord_masks, gm_masks, domains = data
        images = images.to(device)
        spinal_cord_masks = spinal_cord_masks.to(device)
        gm_masks = gm_masks.to(device)
        outputs = best_model(images)
        outputs = F.sigmoid(outputs)

        for i in range(outputs.size(0)):
            prediction = outputs[i].detach().cpu().numpy()
            prediction = np.where(prediction >= 0.5, 1.0, 0.0)
            spinal_cord_mask = spinal_cord_masks[i].detach().cpu().numpy()
            gm_mask = gm_masks[i].detach().cpu().numpy()
            prediction_list_test.append(prediction)
            spinal_cord_mask_list_test.append(spinal_cord_mask)
            gm_mask_list_test.append(gm_mask)

    test_dice_score = metrics.compute_dice(prediction_list_test, spinal_cord_mask_list_test, gm_mask_list_test)
    test_asd_score = metrics.compute_asd(prediction_list_test, spinal_cord_mask_list_test, gm_mask_list_test)

    val_dice_score = torch.load(model_path)["dice"]["data"]
    val_asd_score = torch.load(model_path)["asd"]["data"]

    data = {
        "val": [val_dice_score[0], val_dice_score[1], val_asd_score[0], val_asd_score[1]],
        "test": [test_dice_score[0], test_dice_score[1], test_asd_score[0], test_asd_score[1]]
    }
    df = pd.DataFrame(data)
    df.index = ['SC_DICE', 'GM_DICE', 'SC_ASD', 'GM_ASD']
    return df

    # print("Test: Spinal Cord Dice {:.4f}, GM Dice {:.4f}".format( #, jaccard {:.4f}, {:.4f}, asd {:.4f}, {:.4f}, hd {:.4f}, {:.4f}
    #     dice_score[0], dice_score[1]
    #     # jaccard_score[0], jaccard_score[1],
    #     # asd_score[0], asd_score[1],
    #     # hd_score[0], hd_score[1]
    # ))


def test_all_domains(data_path, batch_size, model_path, encoder, device):
    df_list = []

    for i in range(4):
        domain = "site" + str(i + 1)
        data = dataset.makeDataset(phase='train', path=data_path, specific_domain=domain, transform_train=None,
                                      transform_eval=None)[domain]
        dataloader = DataLoader(data, batch_size, shuffle=False)

        df = do_test(device, domain, dataloader, model_path, encoder, fourier_transform=False)
        df_ft = do_test(device, domain, dataloader, model_path, encoder, fourier_transform=True)

        df.rename(columns={"val": domain + "_val", "test": domain + "_test"}, inplace=True)
        df_ft.rename(columns={"val": "ft_" + domain + "_val", "test": "ft_" + domain + "_test"}, inplace=True)

        df_list.append(df)
        df_list.append(df_ft)

    return pd.concat(df_list, axis=1)
