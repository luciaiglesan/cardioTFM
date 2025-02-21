from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_50epochs(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device='cuda'):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50

    def run_training(self):
        self.print_to_log_file("Custom Trainer: Limiting training to 50 epochs.")
        super().run_training()
