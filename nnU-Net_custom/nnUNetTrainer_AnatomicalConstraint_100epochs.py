import torch 
from torch import nn
import torch.nn.functional as F

import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.helpers import softmax_helper_dim1

from monai.data import decollate_batch, list_data_collate
from monai.transforms import CropForeground, SpatialPad, CenterSpatialCrop


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(

            # S1
            nn.Conv3d(4, 16, kernel_size=3, stride=2, padding=1),       # [4, 24, 128, 128] -> [16, 12, 64, 64]
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            #S2
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),      # [16, 12, 64, 64] -> [32, 6, 32, 32]
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            #S3
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),      # [32, 6, 32, 32] -> [64, 3, 16, 16]
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            #S4
            nn.Conv3d(64, 1, kernel_size=3, stride=(1,2,2), padding=1),  # [64, 3, 16, 16] -> [1, 3, 8, 8]
            nn.BatchNorm3d(1),
            nn.ReLU(),
        )

        # Fully connected layer (Latent space)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64),
            nn.Linear(64, 3 * 8 * 8),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(

            #S4
            nn.ConvTranspose3d(1, 64, kernel_size=3, stride=(1,2,2), 
                               padding=1, output_padding=(0,1,1)),      # [1, 3, 8, 8] -> [64, 3, 16, 16]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            #S3
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),            # [64, 3, 16, 16] -> [32, 6, 32, 32]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            #S2
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, 
                               padding=1, output_padding=1),            # [32, 6, 32, 32] -> [16, 12, 64, 64]
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            #S1
            nn.ConvTranspose3d(16, 16, kernel_size=3, stride=2, 
                               padding=1, output_padding=(0,1,1)),      # [16, 12, 64, 64] -> [16, 24, 128, 128]
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 4, kernel_size=3, padding=1),                 # [16, 24, 128, 128] -> [4, 24, 128, 128]
        )
    
    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)  # [batch_size, 1, 3, 8, 8]
        flattened = encoded.view(encoded.size(0), -1) 
        latent_vector = self.fc[1](flattened)
        # Decoder
        fc_output = self.fc[2:](latent_vector)
        reshaped = fc_output.view(encoded.size())
        decoded = self.decoder(reshaped)
        return decoded, latent_vector

    def compute_embeddings(self, x):
        encoded = self.encoder(x)  # [batch_size, 1, 3, 8, 8]
        return self.fc[1](encoded.view(encoded.size(0), -1))
  


class AnatomicalConstraint_loss(nn.Module):
    def __init__(self, autoencoder):
        super(AnatomicalConstraint_loss, self).__init__()
        self.autoencoder = autoencoder
        self.apply_nonlin = softmax_helper_dim1
        self.crop_foreground = CropForeground()
        self.center_crop = CenterSpatialCrop(roi_size=(24,128,128))
        self.spatial_pad = SpatialPad(spatial_size=(24,128,128))

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
        net_output = torch.argmax(net_output, dim=1, keepdim=True)

        # Transform net_output and target applying CropForeground and SpatialPad
        net_output_list, target_list = [], []

        for net_output_i, target_i in zip(net_output, target):

            foreground_bbox = self.crop_foreground.compute_bounding_box(target_i)
            box_start = np.array([0, foreground_bbox[0][1], foreground_bbox[0][2]])
            box_end = np.array([24, foreground_bbox[1][1], foreground_bbox[1][2]])

            target_i = self.crop_foreground.crop_pad(target_i, box_start, box_end)
            target_i = target_i.to(device='cuda', dtype=torch.float32)
            target_i = self.center_crop(target_i)
            target_i = self.spatial_pad(target_i)
            target_list.append(target_i)

            net_output_i = self.crop_foreground.crop_pad(net_output_i, box_start, box_end)
            net_output_i = net_output_i.to(device='cuda', dtype=torch.float32)
            net_output_i = self.center_crop(net_output_i)
            net_output_i = self.spatial_pad(net_output_i)
            net_output_list.append(net_output_i)
        
        net_output = torch.stack(net_output_list)
        target = torch.stack(target_list)

        # Convert net_output and target to one-hot encoding
        net_output = F.one_hot(net_output.squeeze(1).long(), num_classes=4) \
                    .permute(0, 4, 1, 2, 3).to(torch.float32)
        target = F.one_hot(target.squeeze(1).long(), num_classes=4) \
                    .permute(0, 4, 1, 2, 3).to(torch.float32)
        
        # Compute embeddings
        with torch.no_grad():
            net_output = self.autoencoder.compute_embeddings(net_output)
            target = self.autoencoder.compute_embeddings(target)
        
        # Compute loss
        loss = F.mse_loss(net_output, target)

        return loss
    

class JointLoss(nn.Module):
    def __init__(self, segmentation_loss, anatomical_constraint_loss, weight_segmentation=1, weight_anatomical_constraint=1):
        super(JointLoss, self).__init__()
        self.weight_segmentation = weight_segmentation
        self.weight_anatomical_constraint = weight_anatomical_constraint
        self.segmentation_loss = segmentation_loss
        self.anatomical_constraint_loss = anatomical_constraint_loss

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        segmentation_loss = self.segmentation_loss(net_output, target)
        anatomical_constraint_loss = self.anatomical_constraint_loss(net_output[0], target[0])
        return self.weight_segmentation * segmentation_loss + self.weight_anatomical_constraint * anatomical_constraint_loss


class nnUNetTrainer_AnatomicalConstraint_100epochs(nnUNetTrainer):

    def _build_loss(self):

        # Segmentation loss (Dice + CE)
        segmentation_loss = DC_and_CE_loss(
            soft_dice_kwargs={
                'batch_dice': self.configuration_manager.batch_dice,
                'smooth': 1e-5, 
                'do_bg': False, 
                'ddp': self.is_ddp
            }, 
            ce_kwargs={}, 
            weight_ce=1, 
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label, 
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            # now wrap the loss
            segmentation_loss = DeepSupervisionWrapper(segmentation_loss, weights)

        # Anatomical constraint loss
        # trained_model = torch.load('/mnt/nfs/home/liglesias/cardioTFM/cardioTFM/autoencoder_model.pth')
        #trained_model = torch.load('/mnt/nfs/projects/CARDIO-HULP/SCRATCH_STUDENTS/liglesias/autoencoder_model.pth')
        trained_model = Autoencoder()
        trained_model.load_state_dict(torch.load('/mnt/nfs/home/liglesias/repos/nnUNet/nnunetv2/training/nnUNetTrainer/autoencoder_weights.pth'))
        # trained_model.load_state_dict(torch.load('/mnt/nfs/home/liglesias/repos/nnUNet/nnunetv2/training/nnUNetTrainer/autoencoder_weights_2.pth'))
        #trained_model = torch.load('/mnt/nfs/home/liglesias/repos/nnUNet/nnunetv2/training/nnUNetTrainer/autoencoder_model.pth')
        # trained_model.to(self.device)
        
        # trained_model.to(self.device)
        
        self.num_epochs = 100
        autoencoder_model = Autoencoder()
        autoencoder_model.load_state_dict(trained_model.state_dict())
        autoencoder_model.to(self.device).to(torch.float32)
        autoencoder_model.eval()

        anatomical_constraint_loss = AnatomicalConstraint_loss(autoencoder_model)

        # Loss: Segmentation_loss + (lambda * AnatomicalConstraint_loss)
        return JointLoss(segmentation_loss, anatomical_constraint_loss,
                         weight_anatomical_constraint=0.01) 