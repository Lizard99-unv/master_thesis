import glob
import os
import torch
import torch.nn as nn
import argparse
from util.utils import get_writer, save_checkpoint
from torch.optim.lr_scheduler import StepLR
from models.FFT_block import Decoder
from augmentations.augmentations import dropout_aug_eeg
from util.cal_pearson import l1_loss, pearson_loss, pearson_metric
from util.dataset import RegressionDataset

parser = argparse.ArgumentParser()

parser.add_argument('--epoch',type=int, default=1000)
parser.add_argument('--batch_size',type=int, default=4) #64
parser.add_argument('--win_len',type=int, default = 10)
parser.add_argument('--sample_rate',type=int, default = 64)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--g_con', default=True, help="experiment for within subject")

parser.add_argument('--in_channel', type=int, default=64, help="channel of the input eeg signal")
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--d_inner', type=int, default=1024) 
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_layers',type=int, default=10)
parser.add_argument('--fft_conv1d_kernel', type=tuple,default=(9, 1))
parser.add_argument('--fft_conv1d_padding',type=tuple, default= (4, 0))
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--dropout',type=float,default=0.3)
parser.add_argument('--lamda',type=float,default=0.2)
parser.add_argument('--writing_interval', type=int, default=10)
parser.add_argument('--saving_interval', type=int, default=10)

parser.add_argument('--dataset_folder',type= str, default="", help='write down your absolute path of dataset folder')
parser.add_argument('--split_folder',type= str, default="split_data")
parser.add_argument('--experiment_folder',default=None, help='write down experiment name')

args = parser.parse_args()

 # Set the arameters and device.

input_length = args.sample_rate * args.win_len 
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Provide the path of the dataset.
# which is split already to train, val, test (1:1:1).
data_folder = os.path.join(args.dataset_folder, args.split_folder)
features = ["eeg"] + ["envelope"]

# Create a directory to store (intermediate) results.
result_folder = 'test_results'
if args.experiment_folder is None:
    experiment_folder = "fft_nlayer{}_dmodel{}_nhead{}_win{}".format(args.n_layers, args.d_model, args.n_head, args.win_len)
else: experiment_folder = args.experiment_folder

save_path = os.path.join(result_folder, experiment_folder)
writer = get_writer(result_folder, experiment_folder)

def main():

    # Set the model and optimizer, scheduler.
    model = Decoder(**vars(args)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                betas=(0.9, 0.98),
                                eps=1e-09)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

    # Define train set and loader.
    train_subjects = ['sub-0' + str(x) if x >= 10 else 'sub-00' + str(x) for x in range(1,72)]
    train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features and os.path.basename(x).split("_-_")[1] in train_subjects]
    train_set= RegressionDataset(train_files, input_length, args.in_channel, 'train', args.g_con)
    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size = args.batch_size,
            num_workers = 4,
            sampler = None,
            drop_last=True,
            shuffle=True)

    # Define validation set and loader.
    val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    val_set = RegressionDataset(val_files, input_length, args.in_channel, 'val', args.g_con)
    val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size = 1,
            num_workers = 4,
            sampler = None,
            drop_last=True,
            shuffle=False)

    # Define test set and loader.
    test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
    test_set = RegressionDataset(test_files, input_length, args.in_channel, 'test', args.g_con)
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size = 1,
        num_workers = 4,
        sampler = None,
        drop_last=True,
        shuffle=False)

    # Train the model.
    print('Start')
    for epoch in range(epoch):
        model.train()
        train_loss = 0

        for inputs, labels, sub_id in train_dataloader:
            optimizer.zero_grad()

            inputs = torch.cat((inputs, dropout_aug_eeg(inputs)), dim=0)
            inputs = inputs.to(device)
            labels = torch.cat((labels, labels), dim=0)
            labels = labels.to(device)
            sub_id = torch.cat((sub_id, sub_id), dim=0)
            sub_id = sub_id.to(device)
            outputs = model(inputs, sub_id)

            l_p = pearson_loss(outputs, labels) 
            l_1 = l1_loss(outputs, labels)
            loss = l_p + lamda * l_1
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        if epoch % writing_interval == 0:
            print(f'|-Train-|{epoch}: {train_loss:.3f}')
            # writer.add_losses("Loss", "train",  train_loss, epoch)
            # writer.add_losses("Loss_l1", "train",  train_loss, epoch)

        # Validate the model.
        val_loss = 0
        val_metric = 0
        
        val_test1_metric = 0
        val_test2_metric = 0
        
        val_subjects = {}
        val_subjects['train'] = {}
        val_subjects['test'] = {}
        
        if epoch % writing_interval == 0:

            model.eval()

            with torch.no_grad():
                for val_inputs, val_labels, val_sub_id in val_dataloader:
                    val_inputs = val_inputs.squeeze(0).to(device)
                    val_labels = val_labels.squeeze(0).to(device)
                    val_sub_id = val_sub_id.to(device)

                    val_outputs = model(val_inputs, val_sub_id)
                    val_loss   += pearson_loss(val_outputs, val_labels).mean()
                    val_metric += pearson_metric(val_outputs, val_labels).mean()
                    
                    s_id = int(val_sub_id[0])
                    if s_id in list(range(71)):
                        if s_id in val_subjects['train']:
                            m = torch.cat((val_subjects['train'][s_id], torch.unsqueeze(pearson_metric(val_outputs, val_labels).mean(), dim=0)))
                            val_subjects['train'][s_id] = m
                        else:
                            val_subjects['train'][s_id] = torch.unsqueeze(pearson_metric(val_outputs, val_labels).mean(),dim=0)
                    elif s_id in list(range(71, 85)):
                        if s_id in val_subjects['test']:
                            m = torch.cat((val_subjects['test'][s_id], torch.unsqueeze(pearson_metric(val_outputs, val_labels).mean(), dim=0)))
                            val_subjects['test'][s_id] = m
                        else:
                            val_subjects['test'][s_id] = torch.unsqueeze(pearson_metric(val_outputs, val_labels).mean(),dim=0)

                for value in val_subjects['train'].values():
                    val_test1_metric += value.mean()
                    
                for value in val_subjects['test'].values():
                    val_test2_metric += value.mean()
                            
                val_loss /= len(val_dataloader)
                val_metric /= len(val_dataloader)
                val_metric = val_metric.mean()
        
                
                val_test1_metric /= len(val_subjects['train'])
                val_test2_metric /= len(val_subjects['test'])
                val_test_metric = 2/3 * val_test1_metric + 1/3 * val_test2_metric

                print(f'|-Validation-|{epoch}: {val_loss.mean().item():.3f} {val_metric.item():.3f}')
                print(f'|-Validation-|{epoch}: Test1 - {val_test1_metric:.3f}, Test2 -  {val_test2_metric:.3f}, Final - {val_test_metric:.3f}')
                # writer.add_losses("Loss", "Validation",  val_loss, epoch)
                # writer.add_losses("Pearson", "Validation",  val_metric, epoch)

                # Test the model.
                test_loss = 0
                test_metric = 0
                test_test1_metric = 0
                test_test2_metric = 0
                
                test_subjects = {}
                test_subjects['train'] = {}
                test_subjects['test'] = {}

                for test_inputs, test_labels, test_sub_id in test_dataloader:
                    test_inputs = test_inputs.squeeze(0).to(device)
                    test_labels = test_labels.squeeze(0).to(device)
                    test_sub_id = test_sub_id.to(device)

                    test_outputs = model(test_inputs, test_sub_id)
                    test_loss += pearson_loss(test_outputs, test_labels).mean()
                    test_metric += pearson_metric(test_outputs, test_labels).mean()
                    s_id = int(test_sub_id[0])
                    if s_id in list(range(71)):
                        if s_id in test_subjects['train']:
                            m = torch.cat((test_subjects['train'][s_id], torch.unsqueeze(pearson_metric(test_outputs, test_labels).mean(), dim=0)))
                            test_subjects['train'][s_id] = m
                        else:
                            test_subjects['train'][s_id] = torch.unsqueeze(pearson_metric(test_outputs, test_labels).mean(),dim=0)
                    elif s_id in list(range(71, 85)):
                        if s_id in test_subjects['test']:
                            m = torch.cat((test_subjects['test'][s_id], torch.unsqueeze(pearson_metric(test_outputs, test_labels).mean(), dim=0)))
                            test_subjects['test'][s_id] = m
                        else:
                            test_subjects['test'][s_id] = torch.unsqueeze(pearson_metric(test_outputs, test_labels).mean(),dim=0)
            
                for value in test_subjects['train'].values():
                    test_test1_metric += value.mean()
                    
                for value in test_subjects['test'].values():
                    test_test2_metric += value.mean()
                
                test_loss /= len(test_dataloader)
                test_metric /= len(test_dataloader)
                test_metric = test_metric.mean()  
                test_test1_metric /= len(test_subjects['train'])
                test_test2_metric /= len(test_subjects['test'])
                test_test_metric = 2/3 * test_test1_metric + 1/3 * test_test2_metric
                print(f'|-Test-|{epoch}: {test_loss.mean().item():.3f} {test_metric.item():.3f}')
                print(f'|-Test-|{epoch}: Test1 - {test_test1_metric:.3f}, Test2 -  {test_test2_metric:.3f}, Final - {test_test_metric:.3f}')
                # writer.add_losses("Loss", "Test",  test_loss.mean().item(), epoch)
                # writer.add_losses("Pearson", "Test",  test_metric, epoch)

        if epoch % args.saving_interval == 0:
            learning_rate = print(optimizer.param_groups[0]["lr"])
            save_checkpoint(model, optimizer, learning_rate, epoch, save_path)    

        scheduler.step()


if __name__ == '__main__':
    main()
