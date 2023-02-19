import torch
import argparse
import json

from models.simple_2d_gan import Simple2DGAN
from models.utils import GANLossWithMSE, GeneratorLossWithMSE, DiscriminatorLossWithBCE
from data.data_utils import load_data

if __name__ == "__main__":
    parse=argparse.ArgumentParser(description='Default AE')
    parse.add_argument(
        "-cached_args_file",
        default="cached_args_train_bNFM2DGAN_MSE0_sleep2sF0_v02142023.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    parse.add_argument(
        "-n", default=None, help="IGNORE THIS ARGUMENT - IT'S A (SLURM) BANDAID"
    )
    args = parse.parse_args()

    print("WARNING: UPDATING THE VALUES OF ARGUMENTS FOUND IN ", args.cached_args_file)
    with open(args.cached_args_file, 'r') as infile:
        new_args_dict = json.load(infile)

        new_args_dict["train_data_path"] = new_args_dict["train_data_path"]
        new_args_dict["val_data_path"] = new_args_dict["val_data_path"]
        new_args_dict['hyperparam_save_path'] = new_args_dict["hyperparam_save_path"]
        new_args_dict['save_dir_for_model'] = new_args_dict["save_dir_for_model"]
        new_args_dict['pretrained_file_path'] = new_args_dict["pretrained_file_path"]#'None', 
        new_args_dict['model_type'] = new_args_dict["model_type"]#'Simple2DGAN', 
        new_args_dict['encoder_type'] = new_args_dict["encoder_type"]#'BlockScalableTieredFrequencyEncoder', 
        new_args_dict['gen_downConv1_out_channels'] = int(new_args_dict["gen_downConv1_out_channels"])#247, 
        new_args_dict['gen_downConv1_kernel_size'] = int(new_args_dict["gen_downConv1_kernel_size"])#11, 
        new_args_dict['gen_downConv1HighFreq_dilation'] = int(new_args_dict["gen_downConv1HighFreq_dilation"])#2, 
        new_args_dict['gen_downConv1MedFreq_dilation'] = int(new_args_dict["gen_downConv1MedFreq_dilation"])#4, 
        new_args_dict['gen_downConv1LowFreq_dilation'] = int(new_args_dict["gen_downConv1LowFreq_dilation"])#84, 
        new_args_dict['gen_downConv2_out_channels'] = int(new_args_dict["gen_downConv2_out_channels"])#248, 
        new_args_dict['gen_downConv2_kernel_width'] = int(new_args_dict["gen_downConv2_kernel_width"])#12, 
        new_args_dict['gen_linear_hidden_dim'] = int(new_args_dict["gen_linear_hidden_dim"])#822, 
        new_args_dict['disc_downConv1_out_channels'] = int(new_args_dict["disc_downConv1_out_channels"])#1000, 
        new_args_dict['disc_downConv1_kernel_size'] = int(new_args_dict["disc_downConv1_kernel_size"])#11, 
        new_args_dict['disc_downConv1HighFreq_dilation'] = int(new_args_dict["disc_downConv1HighFreq_dilation"])#2, 
        new_args_dict['disc_downConv1MedFreq_dilation'] = int(new_args_dict["disc_downConv1MedFreq_dilation"])#45, 
        new_args_dict['disc_downConv1LowFreq_dilation'] = int(new_args_dict["disc_downConv1LowFreq_dilation"])#74, 
        new_args_dict['disc_downConv2_out_channels'] = int(new_args_dict["disc_downConv2_out_channels"])#449, 
        new_args_dict['disc_downConv2_kernel_width'] = int(new_args_dict["disc_downConv2_kernel_width"])#8, 
        new_args_dict['disc_linear_hidden_dim'] = int(new_args_dict["disc_linear_hidden_dim"])#560, 
        new_args_dict['gen_num_frequency_conv_blocks'] = int(new_args_dict["gen_num_frequency_conv_blocks"])#2, 
        new_args_dict['gen_num_linear_layers'] = int(new_args_dict["gen_num_linear_layers"])#4, 
        new_args_dict['disc_num_frequency_conv_blocks'] = int(new_args_dict["disc_num_frequency_conv_blocks"])#2, 
        new_args_dict['disc_num_linear_layers'] = int(new_args_dict["disc_num_linear_layers"])#7, 
        new_args_dict['gen_len_temporal_input'] = int(new_args_dict["gen_len_temporal_input"])#1901, 
        new_args_dict['gen_num_out_timesteps'] = int(new_args_dict["gen_num_out_timesteps"])#100, 
        new_args_dict['gen_num_observed_regions'] = int(new_args_dict["gen_num_observed_regions"])#25, 
        new_args_dict['disc_len_temporal_input'] = int(new_args_dict["disc_len_temporal_input"])#2001, 
        new_args_dict['disc_num_observed_regions'] = int(new_args_dict["disc_num_observed_regions"])#25, 
        new_args_dict['disc_linear_out_dim'] = int(new_args_dict["disc_linear_out_dim"])#1, 
        new_args_dict['gen_learning_rate'] = float(new_args_dict["gen_learning_rate"])#0.0001, 
        new_args_dict['disc_learning_rate'] = float(new_args_dict["disc_learning_rate"])#0.0002, 
        new_args_dict['gen_weight_decay'] = float(new_args_dict["gen_weight_decay"])#0.0001, 
        new_args_dict['disc_weight_decay'] = float(new_args_dict["disc_weight_decay"])#0.0001, 
        new_args_dict['gen_eps'] = float(new_args_dict["gen_eps"])#0.0001, 
        new_args_dict['disc_eps'] = float(new_args_dict["disc_eps"])#0.0001, 
        new_args_dict['num_gen_burnin_epochs'] = int(new_args_dict["num_gen_burnin_epochs"])#3, 
        new_args_dict['max_epochs'] = int(new_args_dict["max_epochs"])#3, 
        new_args_dict['num_overfitting_val_trials'] = int(new_args_dict["num_overfitting_val_trials"])#3, 
        new_args_dict['save_freq'] = int(new_args_dict["save_freq"])#1, 
        new_args_dict["batch_size"] = int(new_args_dict["batch_size"])
        new_args_dict["num_system_states"] = int(new_args_dict["num_system_states"])
        new_args_dict["mse_coeff"] = float(new_args_dict["mse_coeff"])
        new_args_dict["disc_coeff"] = float(new_args_dict["disc_coeff"])
        # ---
        new_args_dict['gen_beta_vals'] = (0.9, 0.999)
        new_args_dict['disc_beta_vals'] = (0.9, 0.999)
        # ... add more reassignments as necessary // WARNING: edit this line when running new experiments on SLURM server

    print("INITIALIZING MODEL")
    model = Simple2DGAN(new_args_dict)
    model = model.to(model.device)

    print("LOADING DATA")
    new_args_dict["train_loader"], new_args_dict["val_loader"] = load_data(
        new_args_dict["train_data_path"], 
        new_args_dict["val_data_path"], 
        new_args_dict["gen_num_out_timesteps"], 
        new_args_dict["batch_size"], 
        new_args_dict["num_system_states"], 
        shuffle=True, 
        shuffle_seed=0
    )

    print("ASSUMING LOSS FUNCTION IS VANILLA MSE")
    new_args_dict["gen_burnin_loss_fn"] = GeneratorLossWithMSE(new_args_dict["mse_coeff"])
    new_args_dict["gen_loss_fn"] = GANLossWithMSE(new_args_dict["disc_coeff"], new_args_dict["mse_coeff"])
    new_args_dict["disc_loss_fn"] = DiscriminatorLossWithBCE(new_args_dict["disc_coeff"])

    print("FITTING MODEL TO TRAINING DATA / EVALUATING ON VAL DATA")
    model.fit(new_args_dict)

    print("MAIN FINISHED!!!!")
    pass
