import os
import numpy as np
from tqdm import trange
from datetime import datetime
import monai
from monai.data import DataLoader, decollate_batch, list_data_collate, Dataset, CacheDataset
from tensorboardX import SummaryWriter
from monai.networks.nets import UNet, AutoEncoder, VarAutoEncoder
from monai.transforms import (
    Activations,
    AddChanneld,
    AsChannelFirstd,
    EnsureChannelFirstd,
    EnsureChannelFirst,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    Lambdad,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandRotate90d,
    RandZoomd,
    RandShiftIntensityd,
    RandGaussianSmoothd,
    RandAxisFlipd,
    RandSpatialCropd,
    Rand3DElasticd,
    Resized,
    Spacingd,
    EnsureTyped,
    EnsureType,
    RandRotate90d
)
from monai.visualize import plot_2d_or_3d_image, GradCAM, CAM
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import pandas as pd
import glob as glob
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

# set_determinism(seed=123)

'''***********************************************************************'''


def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return (pvalue_pred)


'''***********************************************************************'''


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy()
    return concordance_index(survtime_all, -hazards, labels)


'''***********************************************************************'''


def MultiLabel_Acc(Pred, Y):
    Pred = Pred.cpu().numpy()
    Y = Y.cpu().numpy()
    acc = None
    for i in range(len(Y[1, :])):
        if i == 0:
            acc = accuracy_score(Y[:, i], Pred[:, i])
        else:
            acc = np.concatenate((acc, accuracy_score(Y[:, i], Pred[:, i])), axis=None)

    return (acc)


'''***********************************************************************'''


def surv_loss(event, time, risk):
    current_batch_len = len(time)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = time[j] >= time[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = event
    theta = risk.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus.float())
    return (loss_nn)


'''***********************************************************************'''


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


'''***********************************************************************'''


def model_eval_gpu(model, datafile, val_transforms):
    data_loader = DataLoader(Dataset(data=datafile, transform=val_transforms), batch_size=16, shuffle=False,
                             num_workers=10, collate_fn=list_data_collate,
                             pin_memory=torch.cuda.is_available())

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSevents = None
        all_OSdurations = None
        all_PFSevents = None
        all_PFSdurations = None
        all_ID = None
        all_OSoutputs = None
        all_PFSoutputs = None
        all_Ageoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSevents = batch_data["OS_status"].cuda()
            OSdurations = batch_data["OS_time"].cuda()
            PFSevents = batch_data["PFS_status"].cuda()
            PFSdurations = batch_data["PFS_time"].cuda()
            # ID = batch_data["ID"].cuda()
            ID_tensor = torch.tensor([int(id_str) for id_str in batch_data["ID"]])
            ID = ID_tensor.cuda()
            OSoutputs, PFSoutputs, Ageoutputs, _ = model(inputs)

            if step == 0:
                all_OSevents = OSevents
                all_OSdurations = OSdurations
                all_PFSevents = PFSevents
                all_PFSdurations = PFSdurations
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
                all_ID = ID
                all_Ageoutputs = Ageoutputs
            else:
                all_OSevents = torch.cat([all_OSevents, OSevents])
                all_OSdurations = torch.cat([all_OSdurations, OSdurations])
                all_PFSevents = torch.cat([all_PFSevents, PFSevents])
                all_PFSdurations = torch.cat([all_PFSdurations, PFSdurations])
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])
                all_ID = torch.cat([all_ID, ID])
                all_Ageoutputs = torch.cat([all_Ageoutputs, Ageoutputs])

            step += 1
        OS_pvalue = cox_log_rank(all_OSoutputs, all_OSevents, all_OSdurations)
        OS_cindex = CIndex_lifeline(all_OSoutputs, all_OSevents, all_OSdurations)
        PFS_pvalue = cox_log_rank(all_PFSoutputs, all_PFSevents, all_PFSdurations)
        PFS_cindex = CIndex_lifeline(all_PFSoutputs, all_PFSevents, all_PFSdurations)

    print(
        f"\n model evaluation"
        f"\n OS c-index: {OS_cindex:.4f} logrank p {OS_pvalue: .4f}"
        f"\n PFS c-index: {PFS_cindex:.4f} logrank p {PFS_pvalue: .4f}"
    )

    return all_OSoutputs, all_PFSoutputs, all_OSevents, all_OSdurations, all_PFSevents, all_PFSdurations, all_ID, all_Ageoutputs


'''***********************************************************************'''


def model_run_gpu(model, datafile, val_transforms):
    data_loader = DataLoader(Dataset(data=datafile, transform=val_transforms), batch_size=100, shuffle=False,
                             num_workers=10, collate_fn=list_data_collate,
                             pin_memory=torch.cuda.is_available())

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSoutputs = None
        all_PFSoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSoutputs, PFSoutputs, _, _ = model(inputs)

            if step == 0:
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs

            else:
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])

            step += 1

    return all_OSoutputs, all_PFSoutputs


'''***********************************************************************'''


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, L0, L1, L2):
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * L0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * L1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * L2 + self.log_vars[2]

        # precision3 = torch.exp(-self.log_vars[3])
        # loss3 = precision3 * L3 + self.log_vars[3]

        # precision4 = torch.exp(-self.log_vars[4])
        # loss4 = precision4 * L4 + self.log_vars[4]

        return loss0 + loss1 + loss2  # + loss3  # + loss4


'''***********************************************************************'''


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

        # self.backbone = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=3, out_channels=1,
        #                                                 dropout_prob =drop_rate)
        self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b1", spatial_dims=3, in_channels=3,
                                                           num_classes=1)
        self.backbone._fc = Identity()
        self.fc1 = torch.nn.Linear(1280, 1)
        self.fc2 = torch.nn.Linear(1280, 1)
        self.fc3 = torch.nn.Linear(1280, 1)
        self.fc4 = torch.nn.Linear(1280, 9)
        # self.fc4 = torch.nn.Linear(1024, 8)

    def forward(self, x):  # define network
        encoded = self.backbone(x)
        os = self.fc1(encoded)
        pfs = self.fc2(encoded)
        age = torch.sigmoid(self.fc3(encoded))
        label = self.fc4(encoded)
        return os, pfs, age, label


'''***********************************************************************'''


def main():
    # hyperparameter setting
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # gpu id
    torch.cuda.empty_cache()
    verbose = False
    model_train = True
    model_pred = True
    model_run = False
    model_visual = True
    BEST_MODEL_NAME = "multitask_subnet1"

    lr = 0.00001
    lr_decay = 0.05
    rand_p = 0.55
    n_loss = 3

    max_epochs = 250  # epoch number
    train_batch = 16
    val_batch = 8
    test_batch = 8
    skip_epoch_model = 50  # not saving the model for initial fluctuation

    print('-------------------start, step 1: load image_precessing data--------------------------')

    train_dir = "/home/medicaldata3/XJ_DATA/DATA_USTC/channal_2_d&v/Input_Train/"
    test_dir = "/home/medicaldata3/XJ_DATA/DATA_USTC/channal_2_d&v/Input_Val/"
    val_dir = "/home/medicaldata3/XJ_DATA/DATA_AYD/ay_box/channal_2_d&v_Resized/"

    train_img = sorted(glob.glob(os.path.join(train_dir, "Patient_*.nii.gz")))  # e.g Patient_0001.nii.gz
    val_img = sorted(glob.glob(os.path.join(val_dir, "Patient_*.nii.gz")))
    test_img = sorted(glob.glob(os.path.join(test_dir, "Patient_*.nii.gz")))

    print("Train Images:")
    print(train_img)
    print("\nValidation Images:")
    print(val_img)
    print("\nTest Images:")
    print(test_img)

    train_os = pd.read_csv(
        "/home/image019/image_003/Sub-network1/Data/ID_STR1/train_events.csv")  # csv file containing ID, OS time and OS events
    val_os = pd.read_csv("/home/image019/image_003/Sub-network1/Data/ID_STR1/valid_events.csv")
    test_os = pd.read_csv("/home/image019/image_003/Sub-network1/Data/ID_STR1/test_events.csv")

    train_os['ID'] = train_os['ID'].astype(str)
    val_os['ID'] = val_os['ID'].astype(str)
    test_os['ID'] = test_os['ID'].astype(str)
    train_os.sort_values('ID')
    val_os.sort_values('ID')
    test_os.sort_values('ID')

    '''--------------reading clinical data including demographics, histological subtypes, metastasis status (Met), stage when IO started (Stage) etc-------------------------------.
    1.OS_status (0,1), 2.OS_time (months), 3.PFS_status (0,1), 4.PFS_time(months), 5.PVTT(Yes,No) 6.LungMet(Yes,No), 7.BoneMet(Yes,No)
    8.Up to seven(Yes,No), 9.LNMet(Yes,No), 10.Age (continuous numbers), 11.Gender(Male,female), 12. Child-Pugh (A,B), 13. HBV_infection (Yes,No), 
    14. BCLC_stage(B,C)
    '''

    train_files = [{"input": in_img, "OS_status": df1, "OS_time": df2, "ID": df3, "PFS_status": df4, "PFS_time": df5,
                    "PVTT": df6, "LungMet": df7, "BoneMet": df8, "Up_to_seven": df9, "LNMet": df10, "Age": df11,
                    "Gender": df12,
                    "Child-Pugh": df13, "HBV_infection": df14, "Stage": df15}  # LN stands for Lymph node
                   for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13,
                       df14, df15 in
                   zip(train_img, train_os["OS_status"], train_os["OS_time"], train_os["ID"], train_os["PFS_status"],
                       train_os["PFS_time"],
                       train_os["PVTT"], train_os["LungMet"], train_os["BoneMet"], train_os["Up_to_seven"],
                       train_os["LNMet"],
                       train_os["Age"], train_os["Male"], train_os["Child-Pugh"], train_os["HBV_infection"],
                       train_os["stage"])]

    val_files = [{"input": in_img, "OS_status": df1, "OS_time": df2, "ID": df3, "PFS_status": df4, "PFS_time": df5,
                  "PVTT": df6, "LungMet": df7, "BoneMet": df8, "Up_to_seven": df9, "LNMet": df10, "Age": df11,
                  "Gender": df12,
                  "Child-Pugh": df13, "HBV_infection": df14, "Stage": df15}  # LN stands for Lymph node
                 for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13,
                     df14, df15 in
                 zip(val_img, val_os["OS_status"], val_os["OS_time"], val_os["ID"], val_os["PFS_status"],
                     val_os["PFS_time"],
                     val_os["PVTT"], val_os["LungMet"], val_os["BoneMet"], val_os["Up_to_seven"],
                     val_os["LNMet"],
                     val_os["Age"], val_os["Male"], val_os["Child-Pugh"], val_os["HBV_infection"], val_os["stage"])]

    test_files = [{"input": in_img, "OS_status": df1, "OS_time": df2, "ID": df3, "PFS_status": df4, "PFS_time": df5,
                   "PVTT": df6, "LungMet": df7, "BoneMet": df8, "Up_to_seven": df9, "LNMet": df10, "Age": df11,
                   "Gender": df12,
                   "Child-Pugh": df13, "HBV_infection": df14, "Stage": df15}  # LN stands for Lymph node
                  for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13,
                      df14, df15 in
                  zip(test_img, test_os["OS_status"], test_os["OS_time"], test_os["ID"], test_os["PFS_status"],
                      test_os["PFS_time"],
                      test_os["PVTT"], test_os["LungMet"], test_os["BoneMet"], test_os["Up_to_seven"],
                      test_os["LNMet"],
                      test_os["Age"], test_os["Male"], test_os["Child-Pugh"], test_os["HBV_infection"],
                      test_os["stage"])]

    print('-------------------start, step 2: define the transforms--------------------------')

    train_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            EnsureChannelFirstd(keys=["input"], channel_dim=0),
            # Resized(keys=["input"], spatial_size=[128, 128, 128]),  # augment, flip, rotate, intensity ...
            RandAdjustContrastd(keys=["input"], prob=rand_p),
            RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
            RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
            RandFlipd(keys=["input"], prob=rand_p),
            RandZoomd(keys="input", prob=rand_p),
            EnsureTyped(keys=["input"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            AsChannelFirstd(keys=["input"], channel_dim=0),
            # Resized(keys=["input"], spatial_size=[128, 128, 128]),
            EnsureTyped(keys=["input"]),
        ]
    )

    print('-------------------start, step 3: define data loader--------------------------')

    # create a training data loader
    train_ds = Dataset(data=train_files, transform=train_transforms)  # , cache_num=50, num_workers=20)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=8,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    for example_data in train_loader:
        print("Shape of 'input' data:", example_data["input"].shape)

    # create a validation data loadertest_ae.py
    val_ds = Dataset(data=val_files, transform=val_transforms)  # , cache_num=40, num_workers=10)
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=3, collate_fn=list_data_collate,
                            pin_memory=True)

    # load the test data
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False, num_workers=3,
                             collate_fn=list_data_collate,
                             pin_memory=True)

    print('-------------------start, step 4: define NET--------------------------')

    # create Net, Loss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet()
    model = model.cuda()
    loss_func = MultiTaskLossWrapper(n_loss).to(device)
    loss_MSE = torch.nn.MSELoss()
    loss_BCE = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay)

    # visualize the NET
    examples = iter(train_loader)
    print(examples)
    example_data = next(examples)
    summary(model, input_size=example_data["input"].shape)
    # writer = SummaryWriter()
    # writer.add_graph(model, example_data["input"].cuda())
    print(model)
    print('-------------------start, step 5: model training--------------------------')

    if model_train:

        time_stamp = "{0:%Y-%m-%d-T%H-%M-%S/}".format(datetime.now()) + BEST_MODEL_NAME
        writer = SummaryWriter(log_dir='runs/' + time_stamp)

        # start a typical PyTorch training
        val_interval = 1
        best_metric_os = -1
        best_metric_pfs = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        val_epoch_loss_values = list()
        test_epoch_loss_values = list()

        t = trange(max_epochs,
                   desc=f"densenet survival -- epoch 0, avg loss: inf", leave=True)

        for epoch in t:
            model.train()
            epoch_loss = 0
            step = 0
            train_OS = None
            train_OStime = None
            train_OSpred = None
            train_PFS = None
            train_PFStime = None
            train_PFSpred = None
            train_label = None
            train_labelpred = None
            t.set_description(f"epoch {epoch + 1} started")

            for batch_data in train_loader:
                inputs = batch_data["input"].cuda()
                t_OS, t_OStime, t_PFS, t_PFStime = batch_data["OS_status"].cuda(), batch_data["OS_time"].cuda(), \
                                                   batch_data["PFS_status"].cuda(), batch_data["PFS_time"].cuda()
                t_Age, t_Gender, t_CP, t_HBV = batch_data["Age"].cuda(), batch_data["Gender"].cuda(), \
                                               batch_data["Child-Pugh"].cuda(), batch_data["HBV_infection"].cuda()
                t_PVTT, t_LMet, t_BoMet, t_Seven, t_LNMet = batch_data["PVTT"].cuda(), batch_data[
                    "LungMet"].cuda(), \
                                                            batch_data["BoneMet"].cuda(), batch_data[
                                                                "Up_to_seven"].cuda(), \
                                                            batch_data["LNMet"].cuda()
                t_stage = batch_data["Stage"].cuda()

                t_Gender, t_CP, t_HBV = t_Gender.unsqueeze(1), t_CP.unsqueeze(1), t_HBV.unsqueeze(1)
                t_PVTT, t_LMet, t_BoMet, t_Seven, t_LNMet = t_PVTT.unsqueeze(1), t_LMet.unsqueeze(1), t_BoMet.unsqueeze(
                    1), t_Seven.unsqueeze(1), t_LNMet.unsqueeze(1)
                t_stage = t_stage.unsqueeze(1)

                t_label = torch.cat((t_Gender, t_CP, t_HBV, t_PVTT, t_LMet, t_BoMet, t_Seven, t_LNMet, t_stage), 1)

                optimizer.zero_grad()

                print("inputs shape", inputs.shape)

                output1, output2, output3, output4 = model(inputs)

                torch.cuda.empty_cache()

                '''**********************Define survival loss*********************************'''
                if step == 0:
                    train_OS = t_OS
                    train_OStime = t_OStime
                    train_OSpred = output1
                    train_PFS = t_PFS
                    train_PFStime = t_PFStime
                    train_PFSpred = output2
                    train_label = t_label
                    train_labelpred = output4
                else:
                    train_OS = torch.cat([train_OS, t_OS])
                    train_OStime = torch.cat([train_OStime, t_OStime])
                    train_OSpred = torch.cat([train_OSpred, output1])
                    train_PFS = torch.cat([train_PFS, t_PFS])
                    train_PFStime = torch.cat([train_PFStime, t_PFStime])
                    train_PFSpred = torch.cat([train_PFSpred, output2])
                    train_label = torch.cat((train_label, t_label), 0)
                    train_labelpred = torch.cat((train_labelpred, output4), 0)

                '''******************add L1 regularization + multitask learning******************'''
                l1_reg = None
                for W in model.parameters():
                    if l1_reg is None:
                        l1_reg = torch.abs(W).sum()
                    else:
                        l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

                t_loss_os = surv_loss(t_OS, t_OStime, output1)
                t_loss_pfs = surv_loss(t_PFS, t_PFStime, output2)

                '''******************define auxiliary loss******************'''
                t_loss_age = loss_MSE(output3, t_Age.unsqueeze(1).float().log_() / 4.75)
                t_loss_label = loss_BCE(output4, t_label.float())
                # t_loss = t_loss_os + w1 * t_loss_pfs + w2 * t_loss_age + w3 * t_loss_label + w4 * l1_reg
                t_loss = loss_func(t_loss_pfs, t_loss_age, t_loss_label) * torch.log10(l1_reg) * t_loss_os
                # t_loss = loss_func(t_loss_os, t_loss_pfs, t_loss_age, t_loss_label) * torch.log10(l1_reg)

                del inputs, t_OS, t_OStime, t_PFS, t_PFStime, t_Age, t_Gender, t_CP, t_HBV, t_PVTT, t_LMet, t_BoMet, t_Seven, t_LNMet
                del t_stage
                if verbose:
                    print(f"\n training epoch: {epoch}, step: {step}")
                    print(f"\n os loss: {t_loss_os:.4f}, pfs loss: {t_loss_pfs:.4f}")
                    print(f"\n age loss: {t_loss_age:.4f}, label loss: {t_loss_label:.4f}")
                    print(f"\n L1 loss: {l1_reg:.4f}, total loss: {t_loss:.4f}")

                torch.cuda.empty_cache()

                step += 1
                t_loss.backward()
                optimizer.step()
                epoch_loss += t_loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train 5 overall loss: step", t_loss.item(), epoch_len * epoch + step)
                writer.add_scalar("train 6 os loss: step", t_loss_os.item(), epoch_len * epoch + step)
                writer.add_scalar("train 7 pfs loss: step", t_loss_pfs.item(), epoch_len * epoch + step)
                writer.add_scalar("train 8 age loss: step", t_loss_age.item(), epoch_len * epoch + step)
                writer.add_scalar("train 10 L1 loss: step", l1_reg.item(), epoch_len * epoch + step)
                writer.add_scalar("train 9 label loss: step", t_loss_label.item(), epoch_len * epoch + step)

            with torch.no_grad():
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                t_pvalue_OS = cox_log_rank(train_OSpred, train_OS, train_OStime)
                t_cindex_OS = CIndex_lifeline(train_OSpred, train_OS, train_OStime)
                writer.add_scalar("train 1 overall log rank OS: epoch", t_pvalue_OS.item(), epoch)
                writer.add_scalar("train 2 overall c-index OS: epoch", t_cindex_OS.item(), epoch)
                t_pvalue_PFS = cox_log_rank(train_PFSpred, train_PFS, train_PFStime)
                t_cindex_PFS = CIndex_lifeline(train_PFSpred, train_PFS, train_PFStime)
                writer.add_scalar("train 3 overall log rank PFS: epoch", t_pvalue_PFS.item(), epoch)
                writer.add_scalar("train 4 overall c-index PFS: epoch", t_cindex_PFS.item(), epoch)

                t_label_pred = (train_labelpred >= 0.)
                t_acc = MultiLabel_Acc(t_label_pred, train_label)
                writer.add_scalar("train 11  Gender accuracy: epoch", t_acc[0].item(), epoch)
                writer.add_scalar("train 12   Child-Pugh accuracy: epoch", t_acc[1].item(), epoch)
                writer.add_scalar("train 13  HBV accuracy: epoch", t_acc[2].item(), epoch)
                writer.add_scalar("train 14  PVTT accuracy: epoch", t_acc[3].item(), epoch)
                writer.add_scalar("train 15  Lung Met accuracy: epoch", t_acc[4].item(), epoch)
                writer.add_scalar("train 16  Bone Met accuracy: epoch", t_acc[5].item(), epoch)
                writer.add_scalar("train 17  Up_to_seven accuracy: epoch", t_acc[6].item(), epoch)
                writer.add_scalar("train 18  LN Met accuracy: epoch", t_acc[7].item(), epoch)
                writer.add_scalar("train 19  stage accuracy: epoch", t_acc[8].item(), epoch)

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_inputs = None
                    val_OS = None
                    val_OStime = None
                    val_OSpred = None
                    val_PFS = None
                    val_PFStime = None
                    val_PFSpred = None
                    val_label = None
                    val_labelpred = None
                    val_epoch_loss = 0
                    val_step = 0
                    for val_data in val_loader:

                        v_inputs, v_OS, v_OStime = val_data["input"].cuda(), val_data["OS_status"].cuda(), val_data[
                            "OS_time"].cuda()
                        v_PFS, v_PFStime = val_data["PFS_status"].cuda(), val_data["PFS_time"].cuda()
                        v_Age, v_Gender, v_CP, v_HBV = val_data["Age"].cuda(), val_data["Gender"].cuda(), \
                                                           val_data["Child-Pugh"].cuda(), val_data["HBV_infection"].cuda()
                        v_PVTT, v_LMet, v_BoMet, v_Seven, v_LNMet = val_data["PVTT"].cuda(), val_data[
                            "LungMet"].cuda(), \
                                                                    val_data["BoneMet"].cuda(), val_data[
                                                                        "Up_to_seven"].cuda(), \
                                                                    val_data["LNMet"].cuda()
                        v_stage = val_data["Stage"].cuda()

                        v_Gender, v_CP, v_HBV = v_Gender.unsqueeze(1), v_CP.unsqueeze(1), v_HBV.unsqueeze(1)
                        v_PVTT, v_LMet, v_BoMet, v_Seven, v_LNMet = v_PVTT.unsqueeze(1), v_LMet.unsqueeze(
                            1), v_BoMet.unsqueeze(1), v_Seven.unsqueeze(1), v_LNMet.unsqueeze(1)
                        v_stage = v_stage.unsqueeze(1)

                        v_label = torch.cat(
                            (v_Gender, v_CP, v_HBV, v_PVTT, v_LMet, v_BoMet, v_Seven, v_LNMet, v_stage), 1)
                        val_output1, val_output2, val_output3, val_output4 = model(v_inputs)

                        '''******************define survival loss******************'''
                        if val_step == 0:
                            val_OS = v_OS
                            val_OStime = v_OStime
                            val_OSpred = val_output1
                            val_PFS = v_PFS
                            val_PFStime = v_PFStime
                            val_PFSpred = val_output2
                            val_label = v_label
                            val_labelpred = val_output4
                        else:
                            val_OS = torch.cat([val_OS, v_OS])
                            val_OStime = torch.cat([val_OStime, v_OStime])
                            val_OSpred = torch.cat([val_OSpred, val_output1])
                            val_PFS = torch.cat([val_PFS, v_PFS])
                            val_PFStime = torch.cat([val_PFStime, v_PFStime])
                            val_PFSpred = torch.cat([val_PFSpred, val_output2])
                            val_label = torch.cat((val_label, v_label), 0)
                            val_labelpred = torch.cat((val_labelpred, val_output4), 0)


                        v_loss_os = surv_loss(v_OS, v_OStime, val_output1)
                        v_loss_pfs = surv_loss(v_PFS, v_PFStime, val_output2)

                        '''******************define auxiliary loss******************'''
                        v_loss_age = loss_MSE(val_output3, v_Age.unsqueeze(1).float().log_() / 4.75)
                        v_loss_label = loss_BCE(val_output4, v_label.float())

                        val_step += 1
                        val_epoch_loss += v_loss_os.item()
                        # v_loss = loss_func(v_loss_os, v_loss_pfs, v_loss_age, v_loss_label)
                        v_loss = loss_func(v_loss_pfs, v_loss_age, v_loss_label) * v_loss_os
                        val_epoch_len = len(val_ds) // val_loader.batch_size + 1
                        writer.add_scalar("valid 5 overall loss: step", v_loss.item(), val_epoch_len * epoch + val_step)
                        writer.add_scalar("valid 6 os loss: step", v_loss_os.item(), val_epoch_len * epoch + val_step)
                        writer.add_scalar("valid 7 pfs loss: step", v_loss_pfs.item(), val_epoch_len * epoch + val_step)
                        writer.add_scalar("valid 8 age loss: step", v_loss_age.item(), val_epoch_len * epoch + val_step)
                        writer.add_scalar("valid 9 label loss: step", v_loss_label.item(),
                                          val_epoch_len * epoch + val_step)

                        torch.cuda.empty_cache()

                    val_epoch_loss /= val_step
                    scheduler.step(val_epoch_loss)
                    val_epoch_loss_values.append(val_epoch_loss)
                    v_pvalue_OS = cox_log_rank(val_OSpred, val_OS, val_OStime)
                    v_cindex_OS = CIndex_lifeline(val_OSpred, val_OS, val_OStime)
                    writer.add_scalar("valid 1 overall log rank OS: epoch", v_pvalue_OS.item(), epoch)
                    writer.add_scalar("valid 2 overall c-index OS: epoch", v_cindex_OS.item(), epoch)
                    writer.add_scalar("learning rate: epoch", optimizer.param_groups[0]['lr'], epoch + 1)
                    v_pvalue_PFS = cox_log_rank(val_PFSpred, val_PFS, val_PFStime)
                    v_cindex_PFS = CIndex_lifeline(val_PFSpred, val_PFS, val_PFStime)
                    writer.add_scalar("valid 3 overall log rank PFS: epoch", v_pvalue_PFS.item(), epoch)
                    writer.add_scalar("valid 4 overall c-index PFS: epoch", v_cindex_PFS.item(), epoch)

                    v_label_pred = (val_labelpred >= 0.)
                    v_acc = MultiLabel_Acc(v_label_pred, val_label)
                    writer.add_scalar("valid 11  Gender accuracy: epoch", v_acc[0].item(), epoch)
                    writer.add_scalar("valid 12  CP accuracy: epoch", v_acc[1].item(), epoch)
                    writer.add_scalar("valid 13  HBV accuracy: epoch", v_acc[2].item(), epoch)
                    writer.add_scalar("valid 14  PVTT accuracy: epoch", v_acc[3].item(), epoch)
                    writer.add_scalar("valid 15  Lung Met accuracy: epoch", v_acc[4].item(), epoch)
                    writer.add_scalar("valid 16  Bone Met accuracy: epoch", v_acc[5].item(), epoch)
                    writer.add_scalar("valid 17  UP_to_seven accuracy: epoch", v_acc[6].item(), epoch)
                    writer.add_scalar("valid 18  LN Met accuracy: epoch", v_acc[7].item(), epoch)
                    writer.add_scalar("valid 19  stage accuracy: epoch", v_acc[8].item(), epoch)

                    '''******************Test the model******************'''
                    test_inputs = None
                    test_OS = None
                    test_OStime = None
                    test_OSpred = None
                    test_PFS = None
                    test_PFStime = None
                    test_PFSpred = None
                    test_label = None
                    test_labelpred = None
                    test_epoch_loss = 0
                    test_step = 0
                    for test_data in test_loader:

                        s_inputs, s_OS, s_OStime = test_data["input"].cuda(), test_data["OS_status"].cuda(), test_data[
                            "OS_time"].cuda()
                        s_PFS, s_PFStime = test_data["PFS_status"].cuda(), test_data["PFS_time"].cuda()
                        s_Age, s_Gender, s_CP, s_HBV = test_data["Age"].cuda(), test_data["Gender"].cuda(), \
                                                           test_data["Child-Pugh"].cuda(), test_data["HBV_infection"].cuda()
                        s_PVTT, s_LMet, s_BoMet, s_Seven, s_LNMet = test_data["PVTT"].cuda(), test_data[
                            "LungMet"].cuda(), \
                                                                    test_data["BoneMet"].cuda(), test_data[
                                                                        "UP_to_seven"].cuda(), \
                                                                    test_data["LNMet"].cuda()
                        s_stage = test_data["Stage"].cuda()

                        s_Gender, s_cp, s_hbv = s_Gender.unsqueeze(1), s_CP.unsqueeze(1), s_HBV.unsqueeze(1)
                        s_PVTT, s_LMet, s_BoMet, s_Seven, s_LNMet = s_PVTT.unsqueeze(1), s_LMet.unsqueeze(
                            1), s_BoMet.unsqueeze(1), s_Seven.unsqueeze(1), s_LNMet.unsqueeze(1)
                        s_stage = s_stage.unsqueeze(1)

                        s_label = torch.cat(
                            (s_Gender, s_CP, s_HBV, s_LMet, s_LMet, s_BoMet, s_Seven, s_LNMet, s_stage), 1)

                        s_output1, s_output2, s_output3, s_output4 = model(s_inputs)

                        '''******************define survival loss******************'''
                        if test_step == 0:
                            test_OS = s_OS
                            test_OStime = s_OStime
                            test_OSpred = s_output1
                            test_PFS = s_PFS
                            test_PFStime = s_PFStime
                            test_PFSpred = s_output2
                            test_label = s_label
                            test_labelpred = s_output4
                        else:
                            test_OS = torch.cat([test_OS, s_OS])
                            test_OStime = torch.cat([test_OStime, s_OStime])
                            test_OSpred = torch.cat([test_OSpred, s_output1])
                            test_PFS = torch.cat([test_PFS, s_PFS])
                            test_PFStime = torch.cat([test_PFStime, s_PFStime])
                            test_PFSpred = torch.cat([test_PFSpred, s_output2])
                            test_label = torch.cat((test_label, s_label), 0)
                            test_labelpred = torch.cat((test_labelpred, s_output4), 0)

                        s_loss_os = surv_loss(s_OS, s_OStime, s_output1)
                        s_loss_pfs = surv_loss(s_PFS, s_PFStime, s_output2)

                        '''*********************define auxiliary loss********************'''
                        s_loss_age = loss_MSE(s_output3, s_Age.unsqueeze(1).float().log_() / 4.75)
                        s_loss_label = loss_BCE(s_output4, s_label.float())

                        test_step += 1
                        test_epoch_loss += s_loss_os.item()
                        # s_loss = loss_func(s_loss_os, s_loss_pfs, s_loss_age, s_loss_label)
                        s_loss = loss_func(s_loss_pfs, s_loss_age, s_loss_label) * s_loss_os
                        test_epoch_len = len(test_ds) // test_loader.batch_size + 1
                        writer.add_scalar("test 5 overall loss: step", s_loss.item(),
                                          test_epoch_len * epoch + test_step)
                        writer.add_scalar("test 6 os loss: step", s_loss_os.item(), test_epoch_len * epoch + test_step)
                        writer.add_scalar("test 7 pfs loss: step", s_loss_pfs.item(),
                                          test_epoch_len * epoch + test_step)
                        writer.add_scalar("test 8 age loss: step", s_loss_age.item(),
                                          test_epoch_len * epoch + test_step)
                        writer.add_scalar("test 9 label loss: step", s_loss_label.item(),
                                          test_epoch_len * epoch + test_step)

                        torch.cuda.empty_cache()

                    test_epoch_loss /= test_step
                    test_epoch_loss_values.append(test_epoch_loss)
                    s_pvalue_OS = cox_log_rank(test_OSpred, test_OS, test_OStime)
                    s_cindex_OS = CIndex_lifeline(test_OSpred, test_OS, test_OStime)
                    writer.add_scalar("test 1 overall log rank OS: epoch", s_pvalue_OS.item(), epoch)
                    writer.add_scalar("test 2 overall c-index OS: epoch", s_cindex_OS.item(), epoch)
                    s_pvalue_PFS = cox_log_rank(test_PFSpred, test_PFS, test_PFStime)
                    s_cindex_PFS = CIndex_lifeline(test_PFSpred, test_PFS, test_PFStime)
                    writer.add_scalar("test 3 overall log rank PFS: epoch", s_pvalue_PFS.item(), epoch)
                    writer.add_scalar("test 4 overall c-index PFS: epoch", s_cindex_PFS.item(), epoch)

                    s_label_pred = (test_labelpred >= 0.)
                    s_acc = MultiLabel_Acc(s_label_pred, test_label)
                    writer.add_scalar("test 11  Gender accuracy: epoch", s_acc[0].item(), epoch)
                    writer.add_scalar("test 12  CP accuracy: epoch", s_acc[1].item(), epoch)
                    writer.add_scalar("test 13  HBV accuracy: epoch", s_acc[2].item(), epoch)
                    writer.add_scalar("test 14  PVTT accuracy: epoch", s_acc[3].item(), epoch)
                    writer.add_scalar("test 15  Lung Met accuracy: epoch", s_acc[4].item(), epoch)
                    writer.add_scalar("test 16  Bone Met accuracy: epoch", s_acc[5].item(), epoch)
                    writer.add_scalar("test 17  UP_to_seven accuracy: epoch", s_acc[6].item(), epoch)
                    writer.add_scalar("test 18  LN Met accuracy: epoch", s_acc[7].item(), epoch)
                    writer.add_scalar("test 19  stage accuracy: epoch", s_acc[8].item(), epoch)

                    metric1 = s_cindex_PFS
                    metric2 = s_cindex_OS
                    if epoch > skip_epoch_model:
                        if metric1 > best_metric_pfs:
                            best_metric_pfs = metric1
                            best_metric_epoch = epoch + 1
                            # os.chdir('runs/' + time_stamp)
                            torch.save(model.state_dict(), BEST_MODEL_NAME + "_PFS.pth")

                        if metric2 > best_metric_os:
                            best_metric_os = metric2
                            torch.save(model.state_dict(), BEST_MODEL_NAME + "_OS.pth")
                            print(f"\n epoch {epoch + 1} saved new best metric model")
                        print(
                            f"\n current epoch: {epoch + 1} current loss: {val_epoch_loss:.4f}"
                            f"\n best loss: {best_metric_pfs:.4f} at epoch {best_metric_epoch}"
                        )

        writer.close()

    print('-------------------start, step 6: model evaluation--------------------------')

    if model_pred:
        # test PFS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_PFS.pth"))

        _, pfs_risk_train, os_train, ostime_train, pfs_train, pfstime_train, ID_train, age_train = model_eval_gpu(model,
                                                                                                                  train_files,
                                                                                                                  val_transforms)
        _, pfs_risk_val, os_val, ostime_val, pfs_val, pfstime_val, ID_val, age_val = model_eval_gpu(model, val_files,
                                                                                                    val_transforms)
        _, pfs_risk_test, os_test, ostime_test, pfs_test, pfstime_test, ID_test, age_test = model_eval_gpu(model,
                                                                                                           test_files,
                                                                                                           val_transforms)

        # test OS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_OS.pth"))
        os_risk_train, _, _, _, _, _, _, _ = model_eval_gpu(model, train_files, val_transforms)
        os_risk_val, _, _, _, _, _, _, _ = model_eval_gpu(model, val_files, val_transforms)
        os_risk_test, _, _, _, _, _, _, _ = model_eval_gpu(model, test_files, val_transforms)

        os_train, ostime_train, pfs_train, pfstime_train, ID_train = os_train.unsqueeze(1), ostime_train.unsqueeze(
            1), pfs_train.unsqueeze(1), pfstime_train.unsqueeze(1), ID_train.unsqueeze(1)
        os_val, ostime_val, pfs_val, pfstime_val, ID_val = os_val.unsqueeze(1), ostime_val.unsqueeze(
            1), pfs_val.unsqueeze(1), pfstime_val.unsqueeze(1), ID_val.unsqueeze(1)
        os_test, ostime_test, pfs_test, pfstime_test, ID_test = os_test.unsqueeze(1), ostime_test.unsqueeze(
            1), pfs_test.unsqueeze(1), pfstime_test.unsqueeze(1), ID_test.unsqueeze(1)

        pred_train_save = torch.cat(
            (os_risk_train, os_train, ostime_train, pfs_risk_train, pfs_train, pfstime_train, ID_train, age_train), 1)
        pred_val_save = torch.cat(
            (os_risk_val, os_val, ostime_val, pfs_risk_val, pfs_val, pfstime_val, ID_val, age_val), 1)
        pred_test_save = torch.cat(
            (os_risk_test, os_test, ostime_test, pfs_risk_test, pfs_test, pfstime_test, ID_test, age_test), 1)

        pred_train_save = pred_train_save.cpu().numpy()
        pred_val_save = pred_val_save.cpu().numpy()
        pred_test_save = pred_test_save.cpu().numpy()

        pred_train_save = pd.DataFrame(pred_train_save)
        pred_val_save = pd.DataFrame(pred_val_save)
        pred_test_save = pd.DataFrame(pred_test_save)

        pred_train_save.to_csv(BEST_MODEL_NAME + "_training.csv")
        pred_val_save.to_csv(BEST_MODEL_NAME + "_validation.csv")
        pred_test_save.to_csv(BEST_MODEL_NAME + "_testing.csv")

    if model_run:
        # test PFS model
        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_PFS.pth"))
        _, pfs_risk = model_run_gpu(model, test_files, val_transforms)

        model.load_state_dict(torch.load(BEST_MODEL_NAME + "_OS.pth"))
        os_risk, _ = model_run_gpu(model, test_files, val_transforms)
        pred_save = torch.cat((os_risk, pfs_risk), 1)
        pred_save = pred_save.cpu().numpy()
        pred_save = pd.DataFrame(pred_save)
        pred_save.to_csv(BEST_MODEL_NAME + "_TCIA.csv")

if __name__ == "__main__":
    main()
