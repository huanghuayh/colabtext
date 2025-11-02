def run():
    import torch
    import numpy as np
    import models.fpn_2 as fpn2
    print('main')
    model = fpn2.PPSP()
    print(torch.cuda.is_available(), np.__version__)
    # print(model)
    with open("rahul_test.txt", "w") as f:
        f.write("my name is Rahul\n")
    # #### load dataset
    # import os, pickle
    #
    # mode = 'gaussian'
    #
    # with open(f"./conv2d_psd_scaled_down_1up_{mode}_1.pkl", "rb") as f:
    #     aggregated_combs_data_lst = pickle.load(f)
    #
    # all_combs_lists = [[0], [3], [0, 1, 2, 3]]
    #
    # print(f"Datasets length={len(aggregated_combs_data_lst)}, combination_length={len(all_combs_lists)}")
    #
    # """
    # train function
    # load model and train
    # """
    # import numpy as np
    # import torch
    # from torch.utils.data import DataLoader, TensorDataset
    # import matplotlib.pyplot as plt
    # import torch.nn as nn
    # psd_length = 1024
    #
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # print(f"Using device: {device}")
    #
    # ### Training function
    # def train_fpn2d_model(X_train, y_train, X_val, y_val, num_epochs=50, batch_size=100, learning_rate=0.001,
    #                       model=None,
    #                       model_name="", train_model_flag=False):
    #     """
    #     Train FPN_2D model with format: [batch_size, 1, channels, 512]
    #     """
    #     ### Convert to tensors
    #     X_train_tensor = torch.FloatTensor(X_train)
    #     y_train_tensor = torch.FloatTensor(y_train)
    #     X_val_tensor = torch.FloatTensor(X_val)
    #     y_val_tensor = torch.FloatTensor(y_val)
    #
    #     print(f"Training data shapes:")
    #     print(f"X_train_tensor: {X_train_tensor.shape}")  # [num_samples, 1, selected_rows, 512]
    #     print(f"y_train_tensor: {y_train_tensor.shape}")  # [num_samples, 512]
    #     print(f"X_val_tensor: {X_val_tensor.shape}")  # [num_samples, 1, selected_rows, 512]
    #     print(f"y_val_tensor: {y_val_tensor.shape}")  # [num_samples, 512]
    #
    #     ### Create datasets and dataloaders
    #     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    #
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #
    #     ### Initialize model, loss, optimizer
    #
    #     class OverlapDiceLoss(nn.Module):
    #         def __init__(self, smooth=1e-6):
    #             super().__init__()
    #             self.smooth = smooth
    #
    #         def forward(self, predictions, targets):
    #
    #             predictions = torch.sigmoid(predictions).float()
    #             targets = targets.float()
    #
    #             # compute per-sample statistics
    #             TP = (predictions * targets).sum(dim=1)
    #             FP = (predictions * (1 - targets)).sum(dim=1)
    #             FN = ((1 - predictions) * targets).sum(dim=1)
    #
    #             # F1 / Dice formulation
    #             dice_per_sample = (2 * TP + self.smooth) / ((2 * TP) + FP + FN + self.smooth)
    #             loss = 1 - dice_per_sample
    #
    #             # mean over batch
    #             return loss.mean()
    #
    #     model.to(device)
    #
    #     # criterion = nn.BCEWithLogitsLoss()
    #     # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    #
    #
    #     criterion = OverlapDiceLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #
    #     if train_model_flag:
    #
    #         # Training variables
    #         train_losses = []
    #         val_losses = []
    #         best_val_loss = float('inf')
    #         best_model_weights = None
    #
    #         ### Training loop
    #         for epoch in range(num_epochs):
    #             ### Training phase
    #             model.train()
    #             train_loss = 0.0
    #
    #             for batch_x, batch_y in train_loader:
    #
    #                 batch_x = batch_x.to(device)  # [batch_size, 1, 2, 512]
    #                 batch_y = batch_y.to(device)  # [batch_size, 512]
    #
    #                 optimizer.zero_grad()
    #                 outputs = model(batch_x)
    #
    #                 ### loss function for model mtl1
    #                 loss_fundamental = criterion(outputs.squeeze(1), batch_y)
    #                 loss = loss_fundamental
    #
    #                 loss.backward()
    #                 optimizer.step()
    #
    #                 train_loss += loss.item() * batch_x.size(0)
    #
    #             train_loss /= len(train_loader.dataset)
    #             train_losses.append(train_loss)
    #
    #             ### Validation phase
    #             model.eval()
    #             val_loss = 0.0
    #
    #             with torch.no_grad():
    #                 for batch_x, batch_y in val_loader:
    #                     batch_x = batch_x.to(device)  # [batch_size, 1, 2, 512]
    #                     batch_y = batch_y.to(device)  # [batch_size, 512]
    #
    #                     outputs = model(batch_x)  # [batch_size, 1, 1, 512]
    #
    #                     ### loss function for model mtl1
    #                     loss_fundamental = criterion(outputs[0].squeeze(1), batch_y)
    #                     loss = loss_fundamental
    #
    #                     val_loss += loss.item() * batch_x.size(0)
    #
    #             val_loss /= len(val_loader.dataset)
    #             val_losses.append(val_loss)
    #
    #             ### Save best model
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 best_model_weights = model.state_dict().copy()
    #                 torch.save(best_model_weights, f'/content/drive/My Drive/train_test_data/{model_name}')
    #
    #             if (epoch + 1) % 50 == 0:
    #                 print(f'Epoch [{epoch + 1}/{num_epochs}], '
    #                       f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
    #                       f'Best Val: {best_val_loss:.4f}')
    #
    #         cur_combination = model_name.split('_')[5].split('.')[0]
    #         # Plot training history
    #         plt.figure(figsize=(10, 5))
    #         plt.plot(train_losses, label='Training Loss')
    #         plt.plot(val_losses, label='Validation Loss')
    #         plt.xlabel('Epochs')
    #         plt.ylabel('Loss')
    #         plt.title('Training and Validation Loss')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.savefig(f'/content/drive/My Drive/train_test_data/training_history_{cur_combination}.png', dpi=300,
    #                     bbox_inches='tight')
    #         plt.close()
    #         # plt.show()
    #
    #     return model
    #
    # for ind, (X_train, X_val, y_train, y_val, dist_train, dist_val) in enumerate(aggregated_combs_data_lst):
    #     if ind == ind:
    #         print(f"Training model: best_fpn1d_model_{mode}_{all_combs_lists[ind]}.pth")
    #
    #         ppsp_model = fpn2.PPSP(in_channels=1)
    #
    #         ### Train the model
    #         trained_model = train_fpn2d_model(
    #             X_train, y_train, X_val, y_val,
    #             num_epochs=150,
    #             batch_size=10,
    #             learning_rate=0.001,
    #             model=ppsp_model,
    #             model_name=f"best_fpn2_1up_model_{mode}_{all_combs_lists[ind]}.pth",
    #             train_model_flag=False
    #         )

if __name__ == '__main__':
    run()