# created 01/20/2023 By Zachary Brown
# references:
#  - UNet Paper: https://arxiv.org/pdf/1505.04597.pdf
#  - Pytorch UNet Implementation: https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
import torch
import numpy as np
import os
import pickle as pkl
import copy
from datetime import datetime
import gc

from models.simple_2d_gan import BlockScalableTieredFrequencyEncoder
from general_utils.plotting import plot_simulation_vs_true_lfp, plot_train_and_val_loss, plot_factor_scores_across_time_steps


class BlockScalableNFMLinearDecoder(torch.nn.Module):
    def __init__(self, num_linear_layers, linear_in_dim, linear_hidden_dim, linear_out_dim):
        super(BlockScalableTieredFrequencyEncoder, self).__init__()
        self.num_linear_layers = num_linear_layers
        self.linear_out_dim = linear_out_dim

        # define block of linear layers
        self.linear_layers = torch.nn.ModuleList() # see https://discuss.pytorch.org/t/append-for-nn-sequential-or-directly-converting-nn-modulelist-to-nn-sequential/7104
        for i in range(num_linear_layers):
            if i == 0:
                self.linear_layers.append(torch.nn.Linear(linear_in_dim, linear_hidden_dim))
                self.linear_layers.append(torch.nn.ReLU())
            elif i == num_linear_layers-1:
                self.linear_layers.append(torch.nn.Linear(linear_hidden_dim, linear_out_dim))
            else:
                self.linear_layers.append(torch.nn.Linear(linear_hidden_dim, linear_hidden_dim))
                self.linear_layers.append(torch.nn.ReLU())
        self.linear_layers = torch.nn.Sequential(*self.linear_layers)
        pass
    
    def forward(self, x):
        # print("BlockScalableNFMLinearDecoder.forward: orig x.size() == ", x.size())
        batch_size = x.size()[0]
        x = torch.flatten(x, start_dim=1)
        # print("BlockScalableNFMLinearDecoder.forward: pre linears x.size() == ", x.size())
        x = self.linear_layers(x)
        # print("BlockScalableNFMLinearDecoder.forward: post linears x.size() == ", x.size())
        return x.view(batch_size, self.linear_out_dim)


class NFM2DGAN(torch.nn.Module):
    """
    Last update: Zac, 02/11/2023

    args keys: 
        gen_len_temporal_input, 
        gen_num_observed_regions, 
        gen_downConv1_out_channels, 
        gen_downConv1_kernel_size, 
        gen_downConv1HighFreq_dilation, 
        gen_downConv1MedFreq_dilation, 
        gen_downConv1LowFreq_dilation, 
        gen_downConv2_out_channels, 
        gen_downConv2_kernel_width, 
        gen_linear_hidden_dim, 
        gen_num_out_timesteps, 
        disc_len_temporal_input, 
        disc_num_observed_regions, 
        disc_downConv1_out_channels, 
        disc_downConv1_kernel_size, 
        disc_downConv1HighFreq_dilation, 
        disc_downConv1MedFreq_dilation, 
        disc_downConv1LowFreq_dilation, 
        disc_downConv2_out_channels, 
        disc_downConv2_kernel_width, 
        disc_linear_hidden_dim, 
        disc_linear_out_dim,
        hyperparam_save_path=None, 
        pretrained_file_path=None, 
        encoder_type="BlockScalableTieredFrequencyEncoder"
        gen_num_frequency_conv_blocks
        disc_num_frequency_conv_blocks
        disc_num_linear_layers
        ---
        gen_encoder_num_linear_layers
        gen_decoder_num_linear_layers
        gen_num_factors_nK

    Current Best Parameter Settings: ???
    """
    def __init__(
            self, 
            args
        ):
        """
        Notes:
         - Estimating NGFM network size:
            * with nK=9, the size of the final layer(s) of the VAE encoder may (roughly) require (25*9)**2 ~ 9.792e-05GB of cuda memory
        """
        super(NFM2DGAN, self).__init__()
        # read in arguments
        gen_len_temporal_input = args["gen_len_temporal_input"]
        gen_num_observed_regions = args["gen_num_observed_regions"]
        gen_downConv1_out_channels = args["gen_downConv1_out_channels"]
        gen_downConv1_kernel_size = args["gen_downConv1_kernel_size"]
        gen_downConv1HighFreq_dilation = args["gen_downConv1HighFreq_dilation"]
        gen_downConv1MedFreq_dilation = args["gen_downConv1MedFreq_dilation"]
        gen_downConv1LowFreq_dilation = args["gen_downConv1LowFreq_dilation"]
        gen_downConv2_out_channels = args["gen_downConv2_out_channels"]
        gen_downConv2_kernel_width = args["gen_downConv2_kernel_width"]
        gen_linear_hidden_dim = args["gen_linear_hidden_dim"]
        gen_num_out_timesteps = args["gen_num_out_timesteps"]
        
        disc_len_temporal_input = args["disc_len_temporal_input"]
        disc_num_observed_regions = args["disc_num_observed_regions"]
        disc_downConv1_out_channels = args["disc_downConv1_out_channels"]
        disc_downConv1_kernel_size = args["disc_downConv1_kernel_size"]
        disc_downConv1HighFreq_dilation = args["disc_downConv1HighFreq_dilation"]
        disc_downConv1MedFreq_dilation = args["disc_downConv1MedFreq_dilation"]
        disc_downConv1LowFreq_dilation = args["disc_downConv1LowFreq_dilation"]
        disc_downConv2_out_channels = args["disc_downConv2_out_channels"]
        disc_downConv2_kernel_width = args["disc_downConv2_kernel_width"]
        disc_linear_hidden_dim = args["disc_linear_hidden_dim"]
        disc_linear_out_dim = args["disc_linear_out_dim"]

        hyperparam_save_path = args["hyperparam_save_path"]
        pretrained_file_path = args["pretrained_file_path"]

        gen_num_factors_nK = args["gen_num_factors_nK"]
        gen_decoder_num_linear_layers = args["gen_decoder_num_linear_layers"]

        # record class member vars
        self.gen_num_factors_nK = gen_num_factors_nK
        self.gen_linear_hidden_dim = gen_linear_hidden_dim
        self.gen_len_temporal_input = gen_len_temporal_input
        self.gen_num_observed_regions = gen_num_observed_regions
        self.gen_num_out_timesteps = gen_num_out_timesteps
        self.pretrained_file_path = pretrained_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Pytorch v0.4.0
        if "encoder_class" in args.keys():
            self.encoder_class = args["encoder_class"]
        else:
            self.encoder_class = "ScalableTieredFrequencyEncoder"

        # construct GAN model architecture
        if self.encoder_class == "BlockScalableTieredFrequencyEncoder":
            gen_num_frequency_conv_blocks = args["gen_num_frequency_conv_blocks"]
            gen_encoder_num_linear_layers = args["gen_encoder_num_linear_layers"]
            disc_num_frequency_conv_blocks = args["disc_num_frequency_conv_blocks"]
            disc_num_linear_layers = args["disc_num_linear_layers"]

            # define generator model
            self.gen_encoder = BlockScalableTieredFrequencyEncoder(
                gen_len_temporal_input, 
                gen_num_observed_regions, 
                gen_num_frequency_conv_blocks, 
                gen_encoder_num_linear_layers, 
                gen_downConv1_out_channels, 
                gen_downConv1_kernel_size, 
                gen_downConv1HighFreq_dilation, 
                gen_downConv1MedFreq_dilation, 
                gen_downConv1LowFreq_dilation, 
                gen_downConv2_out_channels, 
                gen_downConv2_kernel_width, 
                gen_linear_hidden_dim, 
                gen_num_observed_regions*gen_linear_hidden_dim
            )
            self.gen_factor_score_encoder = torch.nn.Sequential( # each dimension of the factor score is gamma_k in paper
                torch.nn.Linear(gen_num_observed_regions*gen_linear_hidden_dim, gen_num_observed_regions*gen_linear_hidden_dim), 
                torch.nn.ReLU(), 
                torch.nn.Linear(gen_num_observed_regions*gen_linear_hidden_dim, self.gen_num_factors_nK)
            )
            self.gen_decoder_factor_modules = torch.nn.ModuleList()
            for _ in range(self.gen_num_factors_nK):
                self.gen_decoder_factor_modules.append(
                    BlockScalableNFMLinearDecoder(gen_decoder_num_linear_layers, gen_linear_hidden_dim, gen_linear_hidden_dim, gen_num_out_timesteps)
                )

            # define discriminator model
            self.disc_model = torch.nn.Sequential(
                BlockScalableTieredFrequencyEncoder(
                    disc_len_temporal_input, 
                    disc_num_observed_regions, 
                    disc_num_frequency_conv_blocks, 
                    disc_num_linear_layers, 
                    disc_downConv1_out_channels, 
                    disc_downConv1_kernel_size, 
                    disc_downConv1HighFreq_dilation, 
                    disc_downConv1MedFreq_dilation, 
                    disc_downConv1LowFreq_dilation, 
                    disc_downConv2_out_channels, 
                    disc_downConv2_kernel_width, 
                    disc_linear_hidden_dim, 
                    disc_num_observed_regions*disc_linear_out_dim
                ), 
                torch.nn.ReLU(), 
                torch.nn.Linear(disc_num_observed_regions, 1), 
                torch.nn.Sigmoid()
            )

        else:
            raise NotImplementedError()
        
        # load any available pretrained weights
        if pretrained_file_path is not None and pretrained_file_path != "None":
            print("NFM2DGAN.__init__: loading parameters from "+str(pretrained_file_path))
            self.load_pretrained_params(pretrained_file_path)
        
        if hyperparam_save_path is not None and hyperparam_save_path != "None":
            print("NFM2DGAN.__init__: SAVING HYPERPARAMS")
            with open(hyperparam_save_path+os.sep+"initial_hyperparams.pkl", 'wb') as outfile:
                pkl.dump({
                    "args": args, 
                }, outfile)
        pass

    def load_pretrained_params(self, pretrained_file_path):
        print("NFM2DGAN.load_pretrained_params: ATTEMPTING TO LOAD WARM-START PARAMS")
        # see https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        self.load_state_dict(torch.load(pretrained_file_path))
        print("NFM2DGAN.load_pretrained_params: SUCCESSFULLY LOADED WARM-START PARAMS")
        pass

    def forward(self, x):
        z = self.gen_encoder(x).view(x.size()[0], self.gen_num_observed_regions, self.gen_linear_hidden_dim)
        gamma = self.gen_factor_score_encoder(torch.flatten(z, start_dim=1))
        x_hat = gamma[:,0][:, None, None]*self.gen_decoder_factor_modules[0](z) # for [:, None, None] syntax see https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        # print("NFM2DGAN.forward: x_hat.size() == ", x_hat.size())
        for k in range(1, self.gen_num_factors_nK):
            # print("NFM2DGAN.forward: k == ", k)
            x_hat += gamma[:,k][:, None, None]*self.gen_decoder_factor_modules[k](z)
        return x_hat, gamma

    def run_simulation(self, x, num_sim_steps=1):
        batch_size = x.size()[0]
        x_simulations = []
        gammas = []
        in_x = copy.deepcopy(x)
        for sim_step in range(num_sim_steps):
            curr_sim_start = sim_step*self.gen_num_out_timesteps
            if sim_step > 0:
                in_x = torch.cat([in_x, x_simulations[-1]], dim=2).view(batch_size, self.gen_num_observed_regions, -1)
            curr_x_hat, curr_gamma = self.forward(in_x[:, :, curr_sim_start:curr_sim_start+self.gen_len_temporal_input])
            curr_x_hat = curr_x_hat.view(-1, self.gen_num_observed_regions, self.gen_num_out_timesteps)
            x_simulations.append(curr_x_hat)
            for _ in range(self.gen_num_out_timesteps):
                gammas.append(curr_gamma.view(batch_size, self.num_factors_nK, 1))
        x_sim = torch.stack(x_simulations, dim=2).view(batch_size, self.gen_num_observed_regions, num_sim_steps*self.gen_num_out_timesteps)
        gamma_sim = torch.stack(gammas, dim=2).view(batch_size, self.gen_num_factors_nK, num_sim_steps*self.gen_num_out_timesteps)
        return x_sim, gamma_sim

    def fit(self, args):
        train_loader = args["train_loader"]
        val_loader = args["val_loader"]
        gen_loss_fn = args["gen_loss_fn"]
        disc_loss_fn = args["disc_loss_fn"]
        gen_burnin_loss_fn = args["gen_burnin_loss_fn"]
        gen_beta_vals = args["gen_beta_vals"]
        disc_beta_vals = args["disc_beta_vals"]
        save_dir_for_model = args["save_dir_for_model"]
        gen_learning_rate = args["gen_learning_rate"]
        disc_learning_rate = args["disc_learning_rate"]
        gen_weight_decay = args["gen_weight_decay"]
        disc_weight_decay = args["disc_weight_decay"]
        gen_eps = args["gen_eps"]
        disc_eps = args["disc_eps"]
        num_gen_burnin_epochs = args["num_gen_burnin_epochs"]
        max_epochs = args["max_epochs"]
        num_overfitting_val_trials = args["num_overfitting_val_trials"]
        save_freq = args["save_freq"]
        
        curr_datetime = datetime.now().isoformat() # see https://www.geeksforgeeks.org/get-current-date-and-time-using-python/ and  https://www.programiz.com/python-programming/datetime/current-datetime
        save_dir_for_model = save_dir_for_model+os.sep+str(curr_datetime)

        print("NFM2DGAN.fit: START OF TRAINING")
        # initialize training state
        saved_model = None
        gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=gen_learning_rate, betas=gen_beta_vals, eps=gen_eps, weight_decay=gen_weight_decay)
        disc_optimizer = torch.optim.Adam(self.disc_model.parameters(), lr=disc_learning_rate, betas=disc_beta_vals, eps=disc_eps, weight_decay=disc_weight_decay)

        # initialize historical loss tracking
        avg_train_gen_losses = []
        avg_train_gen_disc_losses = []
        avg_train_gen_MSE_losses = []
        avg_val_gen_losses = []
        avg_val_gen_disc_losses = []
        avg_val_gen_MSE_losses = []
        avg_train_disc_losses = []
        avg_val_disc_losses = []

        # Iterate over epochs
        # print("<<< TRAINING START >>>")
        min_avg_val_loss = np.inf
        for epoch in range(max_epochs):
            print("epoch == ", epoch)
            
            # initialize batch loss tracking
            running_train_gen_loss = 0
            running_train_gen_disc_loss = 0
            running_train_gen_MSE_loss = 0
            running_val_gen_loss = 0
            running_val_gen_disc_loss = 0
            running_val_gen_MSE_loss = 0
            running_train_disc_loss = 0
            running_val_disc_loss = 0
            
            # prep Discriminator model (disc_model) for updating/training weights
            self.disc_model.train()
            self.gen_model.eval()
        
            # iterate over training batches and update Discriminator (disc_model) weights
            for batch_num, (x, _) in enumerate(train_loader):
                # print("discriminator update batch == ", batch_num)
                curr_batch_size = x.size()[0]
                y = torch.zeros(curr_batch_size*2).float().view(-1)
                # mark the first curr_batch_size elements as being real
                y[:curr_batch_size] += 1.
                # transfer to GPU/device
                x, y = x.to(self.device), y.to(self.device)
                # zero out any pre-existing gradients
                disc_optimizer.zero_grad()

                # make prediction(s)
                x_sims, _ = self.run_simulation(x[:,:,:self.gen_len_temporal_input], num_sim_steps=1)
                assert x_sims.size() == x[:, :, self.gen_len_temporal_input:].size()

                # combine real and synthetic samples for discriminator
                rand_inds = torch.randperm(y.size()[0])
                y = y[rand_inds]
                x_combined = torch.cat((x[:, :, :], torch.cat((x[:, :, :self.gen_len_temporal_input], x_sims), dim=2)), dim=0)[rand_inds,:,:]

                #compute resulting loss
                disc_preds_b1 = self.disc_model(x_combined[:curr_batch_size,:,:])
                disc_preds_b2 = self.disc_model(x_combined[curr_batch_size:,:,:])
                curr_disc_loss = disc_loss_fn(disc_preds_b1.squeeze(), y[:curr_batch_size]) + disc_loss_fn(disc_preds_b2.squeeze(), y[curr_batch_size:])
                
                # update weights
                curr_disc_loss.backward()
                disc_optimizer.step()

                # track loss
                running_train_disc_loss += curr_disc_loss

                # free up cuda memory
                del x
                del y
                del x_sims
                del x_combined
                del rand_inds
                del disc_preds_b1
                del disc_preds_b2
                # torch.cuda.empty_cache()
                # print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING
                # break # FOR DEBUGGING
            
            # gc.collect()
            # torch.cuda.empty_cache()

            # prep generative model for updating/training weights
            self.disc_model.eval()
            self.gen_model.train()
        
            # iterate over training batches and update NFM (gen_model) weights
            for batch_num, (x, y) in enumerate(train_loader):
                # print("generator update batch == ", batch_num)
                # transfer to GPU/device
                x, y = x.to(self.device), y.to(self.device)
                curr_disc_target_labels = torch.zeros(x.size()[0]).float().to(self.device)
                # zero out any pre-existing gradients
                gen_optimizer.zero_grad()

                # make prediction(s)
                if epoch < num_gen_burnin_epochs:
                    x_sims, _ = self.run_simulation(x[:,:,:self.gen_len_temporal_input], num_sim_steps=1)
                    assert x_sims.size() == x[:, :, self.gen_len_temporal_input:self.gen_len_temporal_input+(1*self.gen_num_out_timesteps)].size()
                    #compute resulting loss
                    p_x_sims = None
                    curr_gen_loss = gen_burnin_loss_fn(x_sims[:, :, :self.gen_num_out_timesteps], x[:, :, self.gen_len_temporal_input:self.gen_len_temporal_input+(1*self.gen_num_out_timesteps)])
                    curr_gen_disc_loss_val = 0
                    curr_gen_mse_loss_val = curr_gen_loss.detach().item()
                else:
                    x_sims, _ = self.run_simulation(x[:,:,:self.gen_len_temporal_input], num_sim_steps=1)
                    assert x_sims.size() == x[:, :, self.gen_len_temporal_input:].size()
                    #compute resulting loss
                    p_x_sims = self.disc_model(torch.cat((x[:, :, :self.gen_len_temporal_input], x_sims), dim=2))
                    curr_gen_loss, (curr_gen_disc_loss_val, curr_gen_mse_loss_val) = gen_loss_fn(
                        x_sims, 
                        x[:, :, self.gen_len_temporal_input:], 
                        p_x_sims.squeeze(), 
                        curr_disc_target_labels
                    )

                # update weights
                curr_gen_loss.backward()
                gen_optimizer.step()

                # track loss
                running_train_gen_loss += curr_gen_loss.detach().item()
                running_train_gen_disc_loss += curr_gen_disc_loss_val
                running_train_gen_MSE_loss += curr_gen_mse_loss_val

                # free up cuda memory
                del x
                del y
                del x_sims
                del curr_disc_target_labels
                del p_x_sims
                # torch.cuda.empty_cache()
                # print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING
                # break # FOR DEBUGGING
            
            # gc.collect()
            # torch.cuda.empty_cache()
            
            # perform validation
            # print("validating")
            with torch.no_grad():
                # prep models for evaluation
                self.disc_model.eval()
                self.gen_model.eval()
                
                # iterate over evlauation batches
                for batch_num, (x, y) in enumerate(val_loader):
                    # print("validation batch == ", batch_num)
                    curr_batch_size = x.size()[0]
                    y_disc = torch.zeros(curr_batch_size*2).float().view(-1)
                    # mark the first curr_batch_size elements as being real
                    y_disc[:curr_batch_size] += 1.
                    # transfer to GPU/device
                    x, y, y_disc = x.to(self.device), y.to(self.device), y_disc.to(self.device)

                    # evaluate generative model
                    x_sims, _ = self.run_simulation(x[:,:,:self.gen_len_temporal_input], num_sim_steps=1)
                    assert x_sims.size() == x[:, :, self.gen_len_temporal_input:].size()
                    p_x_sims = self.disc_model(torch.cat((x[:, :, :self.gen_len_temporal_input], x_sims), dim=2))
                    
                    # compute resulting loss
                    if epoch < num_gen_burnin_epochs:
                        curr_gen_loss = gen_burnin_loss_fn(x_sims[:, :, :self.gen_num_out_timesteps], x[:, :, self.gen_len_temporal_input:self.gen_len_temporal_input+(1*self.gen_num_out_timesteps)])
                        curr_gen_disc_loss_val = 0
                        curr_gen_mse_loss_val = curr_gen_loss.detach().item()
                    else:
                        curr_gen_loss, (curr_gen_disc_loss_val, curr_gen_mse_loss_val) = gen_loss_fn(
                            x_sims, 
                            x[:, :, self.gen_len_temporal_input:], 
                            p_x_sims.squeeze(), 
                            y_disc[curr_batch_size:]#curr_disc_target_labels
                        )
                    running_val_gen_loss += curr_gen_loss.detach().item()
                    running_val_gen_disc_loss += curr_gen_disc_loss_val
                    running_val_gen_MSE_loss += curr_gen_mse_loss_val

                    # evaluate discriminator model
                    x_combined = torch.cat((x[:, :, :], torch.cat((x[:, :, :self.gen_len_temporal_input], x_sims), dim=2)), dim=0)
                    disc_preds_b1 = self.disc_model(x_combined[:curr_batch_size,:,:])
                    disc_preds_b2 = self.disc_model(x_combined[curr_batch_size:,:,:])
                    curr_disc_loss = disc_loss_fn(disc_preds_b1.squeeze(), y_disc[:curr_batch_size]) + disc_loss_fn(disc_preds_b2.squeeze(), y_disc[curr_batch_size:])
                    running_val_disc_loss += curr_disc_loss.detach().item()

                    # free up cuda memory
                    del x
                    del y
                    del y_disc
                    del x_sims
                    del x_combined
                    del disc_preds_b1
                    del disc_preds_b2
                    # torch.cuda.empty_cache()
                    # print("BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING
                    # break # FOR DEBUGGING
                pass
            
            gc.collect()
            torch.cuda.empty_cache()

            # record averages
            avg_train_gen_losses.append(running_train_gen_loss / len(train_loader))
            avg_train_gen_disc_losses.append(running_train_gen_disc_loss / len(train_loader))
            avg_train_gen_MSE_losses.append(running_train_gen_MSE_loss / len(train_loader))
            avg_val_gen_losses.append(running_val_gen_loss / len(val_loader))
            avg_val_gen_disc_losses.append(running_val_gen_disc_loss / len(val_loader))
            avg_val_gen_MSE_losses.append(running_val_gen_MSE_loss / len(val_loader))
            avg_train_disc_losses.append(running_train_disc_loss / len(train_loader))
            avg_val_disc_losses.append(running_val_disc_loss / len(val_loader))
            
            # check stopping criterion / save model
            if epoch >= num_gen_burnin_epochs:
                if avg_val_gen_disc_losses[-1] <= min_avg_val_loss: # best val performance encountered so far
                    min_avg_val_loss = avg_val_gen_disc_losses[-1]
                    saved_model = self.state_dict()
                elif sum([avg_val_gen_disc_losses[-1] > past_avg for past_avg in avg_val_gen_disc_losses[-(num_overfitting_val_trials+1):-1]]) == num_overfitting_val_trials: # case where we've reached our overfitting stopping critereia
                    print("NFM2DGAN.train: EARLY STOPPING on epoch ", epoch)
                    break
            
            # save intermediate state_dicts just in case
            if epoch % save_freq == 0:
                if epoch == 0:
                    os.mkdir(save_dir_for_model) # see https://www.geeksforgeeks.org/create-a-directory-in-python/
                temp_model_save_path = os.path.join(save_dir_for_model, "temp_model_epoch"+str(epoch)+".bin")
                torch.save(saved_model, temp_model_save_path)
                plot_train_and_val_loss(avg_train_gen_losses, avg_val_gen_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_loss_visualization_epoch"+str(epoch)+".png")
                plot_train_and_val_loss(avg_train_gen_disc_losses, avg_val_gen_disc_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_disc_loss_visualization_epoch"+str(epoch)+".png")
                plot_train_and_val_loss(avg_train_gen_MSE_losses, avg_val_gen_MSE_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_MSE_loss_visualization_epoch"+str(epoch)+".png")
                plot_train_and_val_loss(avg_train_disc_losses, avg_val_disc_losses, save_dir_for_model+os.sep+"discriminator_avg_train_vs_val_loss_visualization_epoch"+str(epoch)+".png")
                # plot simulations of signals
                for i in [10, 11, 12]:
                    print("\tNow running simulation: ", i, " of ", len([10, 11, 12]))
                    x_train, y_train = train_loader.dataset[i]
                    x_train, y_train = x_train.to(self.device).view(1, 25, -1), y_train.to(self.device)
                    x_val, y_val = val_loader.dataset[i]
                    x_val, y_val = x_val.to(self.device).view(1, 25, -1), y_val.to(self.device)

                    # get items for plotting
                    x_train_simulation, gamma_train_simulation = self.run_simulation(x_train[:,:,:self.gen_len_temporal_input], 1)
                    x_val_simulation, gamma_val_simulation = self.run_simulation(x_val[:,:,:self.gen_len_temporal_input], 1)

                    # determine current transition type (global sleep state / label)
                    train_transit_type = str(torch.argmax(y_train[:,0]).detach().item()) + "-to-" + str(torch.argmax(y_train[:,-1]).detach().item())
                    val_transit_type = str(torch.argmax(y_val[:,0]).detach().item()) + "-to-" + str(torch.argmax(y_val[:,-1]).detach().item())

                    # plot gamma score results
                    plot_factor_scores_across_time_steps(gamma_train_simulation, save_dir_for_model+os.sep+"epoch"+str(epoch)+"_train_simulation"+str(i)+"_transitLabel"+train_transit_type+"_GAMMA_SCORE_visualization.png")
                    plot_factor_scores_across_time_steps(gamma_val_simulation, save_dir_for_model+os.sep+"epoch"+str(epoch)+"_val_simulation"+str(i)+"_transitLabel"+val_transit_type+"_GAMMA_SCORE_visualization.png")

                    # plot time series predictions
                    for chan_ind in range(x_train.size()[1]//4):
                        print("\t\t plotting simulation for chan_ind == ", chan_ind)
                        plot_simulation_vs_true_lfp(
                            x_train[0, chan_ind, :self.gen_len_temporal_input].cpu().detach().numpy(), 
                            x_train_simulation[0, chan_ind, :].cpu().detach().numpy(), 
                            x_train[0, chan_ind, self.gen_len_temporal_input:].cpu().detach().numpy(), 
                            save_dir_for_model+os.sep+"epoch"+str(epoch)+"_train_simulation"+str(i)+"_channel"+str(chan_ind)+"_transitLabel"+train_transit_type+"_visualization.png"
                        )
                        plot_simulation_vs_true_lfp(
                            x_val[0, chan_ind, :self.gen_len_temporal_input].cpu().detach().numpy(), 
                            x_val_simulation[0, chan_ind, :].cpu().detach().numpy(), 
                            x_val[0, chan_ind, self.gen_len_temporal_input:].cpu().detach().numpy(), 
                            save_dir_for_model+os.sep+"epoch"+str(epoch)+"_val_simulation"+str(i)+"_channel"+str(chan_ind)+"_transitLabel"+val_transit_type+"_visualization.png"
                        )
                        pass
                    del x_train
                    del y_train
                    del x_val
                    del y_val
                    del x_train_simulation
                    del x_val_simulation
                    del gamma_train_simulation
                    del gamma_val_simulation
                    pass
                torch.cuda.empty_cache()
            
            # if epoch >= (num_gen_burnin_epochs + 1): # FOR DEBUGGING
            #     print("BREAKING EPOCH LOOP FOR DEBUGGING PURPOSES\n\n") # FOR DEBUGGING
            #     break # FOR DEBUGGING

        # print("NFM2DGAN.train: END OF TRAINING - now saving final model / other info")

        # save final model
        model_save_path = os.path.join(save_dir_for_model, "final_trained_model.bin")
        torch.save(saved_model, model_save_path)

        plot_train_and_val_loss(avg_train_gen_losses, avg_val_gen_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_loss_visualization_FINAL.png")
        plot_train_and_val_loss(avg_train_gen_disc_losses, avg_val_gen_disc_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_disc_loss_visualization_FINAL.png")
        plot_train_and_val_loss(avg_train_gen_MSE_losses, avg_val_gen_MSE_losses, save_dir_for_model+os.sep+"generator_avg_train_vs_val_MSE_loss_visualization_FINAL.png")
        plot_train_and_val_loss(avg_train_disc_losses, avg_val_disc_losses, save_dir_for_model+os.sep+"discriminator_avg_train_vs_val_loss_visualization_FINAL.png")
        
        meta_data_save_path = os.path.join(save_dir_for_model, "training_meta_data_and_hyper_parameters.pkl")
        with open(meta_data_save_path, "wb") as outfile:
            pkl.dump({
                "max_epochs": max_epochs, 
                "num_overfitting_val_trials": num_overfitting_val_trials, 
                "save_freq": save_freq, 
                "num_gen_burnin_epochs": num_gen_burnin_epochs,
                "gen_learning_rate": gen_learning_rate, 
                "gen_beta_vals": gen_beta_vals, 
                "gen_weight_decay": gen_weight_decay, 
                "gen_eps": gen_eps, 
                "avg_train_gen_losses": avg_train_gen_losses, 
                "avg_train_gen_disc_losses": avg_train_gen_disc_losses,
                "avg_train_gen_MSE_losses": avg_train_gen_MSE_losses, 
                "avg_val_gen_losses": avg_val_gen_losses, 
                "avg_val_gen_disc_losses": avg_val_gen_disc_losses, 
                "avg_val_gen_MSE_losses": avg_val_gen_MSE_losses, 
                "disc_learning_rate": disc_learning_rate, 
                "disc_beta_vals": disc_beta_vals, 
                "disc_weight_decay": disc_weight_decay, 
                "disc_eps": disc_eps, 
                "avg_train_disc_losses": avg_train_disc_losses, 
                "avg_val_disc_losses": avg_val_disc_losses,  
            }, outfile)
        
        print("NFM2DGAN.train: DONE!")
        return avg_train_gen_losses, avg_train_gen_disc_losses, avg_train_gen_MSE_losses, avg_val_gen_losses, avg_val_gen_disc_losses, avg_val_gen_MSE_losses, avg_train_disc_losses, avg_val_disc_losses
