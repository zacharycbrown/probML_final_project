import torch
import torch.nn as nn
# from torch_geometric.nn import GCNConv # DUKE DCC PYTHON PACKAGES DO NOT CURRENTLY SUPPORT torch-geometric
import torch.nn.functional as F
import math

# Basic DNN Architectures ==================================================================
class BasicEncoder(nn.Module):
    def __init__(self, num_input_time_steps, num_channels, latent_node_state_dim, num_temporal_latent_chans, initial_temporal_kernel_size, 
                       final_temperal_kernel_size, final_temperal_kernel_stride, pre_linear_dim):
        super(BasicEncoder, self).__init__()
        self.num_input_time_steps = num_input_time_steps
        self.num_channels = num_channels
        self.latent_node_state_dim = latent_node_state_dim
        self.num_temporal_latent_chans = num_temporal_latent_chans
        self.spacial_kernel_size = num_channels+(1-(num_channels%2))
        self.initial_temporal_kernel_size = initial_temporal_kernel_size # SMALL TEMPORAL CONV LAYER
        self.final_temperal_kernel_size = final_temperal_kernel_size
        self.final_temperal_kernel_stride = final_temperal_kernel_stride
        self.pre_linear_dim = pre_linear_dim

        # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
        self.temporal_conv_layers = nn.Sequential( 
            nn.Conv1d(num_channels, num_temporal_latent_chans, initial_temporal_kernel_size, stride=1, padding=initial_temporal_kernel_size//2, dilation=1, padding_mode='circular'), 
            nn.ReLU(), 
            nn.Conv1d(num_temporal_latent_chans, num_channels, final_temperal_kernel_size, stride=final_temperal_kernel_stride, padding=final_temperal_kernel_size//2, dilation=1, padding_mode='circular')
        )
        self.spacial_conv_layer = nn.Sequential( 
            nn.Conv1d(pre_linear_dim, pre_linear_dim, self.spacial_kernel_size, stride=1, padding=num_channels//2, dilation=1, padding_mode='circular'),#padding=num_channels-1, dilation=1, padding_mode='zeros'), 
            nn.ReLU(), 
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(num_channels*pre_linear_dim, num_channels*latent_node_state_dim), 
            nn.ReLU(), 
            nn.Linear(num_channels*latent_node_state_dim, num_channels*latent_node_state_dim), 
        )
        pass
    
    def forward(self, x):
        x = self.temporal_conv_layers(x)
        x = torch.transpose(x, 1, 2) # prepare for spacial convolution
        x = self.spacial_conv_layer(x)
        x = torch.transpose(x, 1, 2) # swap channel and temporal dim in preparation for temporal convolutions
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layer(x)
        return x.view(x.size()[0], self.num_channels, self.latent_node_state_dim)


class BasicDecoder(nn.Module):
    def __init__(self, num_input_features, num_channels, num_pred_time_steps, hidden_dim):
        super(BasicDecoder, self).__init__()
        self.num_input_features = num_input_features
        self.num_channels = num_channels
        self.num_pred_time_steps = num_pred_time_steps
        self.hidden_dim = hidden_dim

        self.linear_block = nn.Sequential(
            nn.Linear(num_channels*num_input_features, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, num_channels*num_pred_time_steps), 
        )
        pass
    
    def forward(self, x):
        # print("BasicDecoder.forward: 1st x.size() == ", x.size())
        x = self.linear_block(x)
        # print("BasicDecoder.forward: returning x.size() == ", x.size())
        return x.view(x.size()[0], self.num_channels, self.num_pred_time_steps)


class BasicConv1dEncoder(nn.Module):
    def __init__(self, num_input_time_steps, num_channels, latent_node_state_dim):
        super(BasicConv1dEncoder, self).__init__()
        self.num_input_time_steps = num_input_time_steps
        self.num_channels = num_channels
        self.latent_node_state_dim = latent_node_state_dim

        self.spacial_kernel_size = num_channels+(1-(num_channels%2))
        if latent_node_state_dim == 56: # case for nC=4, nK=1, nF=10
            self.initial_temporal_kernel_size = 10 # SMALL TEMPORAL CONV LAYER
            self.final_temperal_kernel_size = (num_input_time_steps//self.initial_temporal_kernel_size)-latent_node_state_dim + (1-latent_node_state_dim%2) # SMALL TEMPORAL CONV LAYE
            # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
            self.temporal_conv_layers = nn.Sequential( 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=self.initial_temporal_kernel_size, padding=0, dilation=1),#, padding_mode='zeros'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.final_temperal_kernel_size, stride=1, padding=0, dilation=1),#, padding_mode='zeros')
            )
        elif latent_node_state_dim == 504: # case for nC=4, nK=9, nF=10
            self.initial_temporal_kernel_size = 3 # SMALL TEMPORAL CONV LAYER
            self.final_temperal_kernel_size = (num_input_time_steps//self.initial_temporal_kernel_size)-latent_node_state_dim + (1-latent_node_state_dim%2) # SMALL TEMPORAL CONV LAYER
            # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
            self.temporal_conv_layers = nn.Sequential( 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=self.initial_temporal_kernel_size, padding=0, dilation=1),#, padding_mode='zeros'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.final_temperal_kernel_size, stride=1, padding=0, dilation=1),#, padding_mode='zeros')
            )
        elif latent_node_state_dim == 514: # case for nC=25, nK=9, nF=514
            self.initial_temporal_kernel_size = 101 # SMALL TEMPORAL CONV LAYER
            self.final_temperal_kernel_size = 61 # SMALL TEMPORAL CONV LAYER
            # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
            self.temporal_conv_layers = nn.Sequential( 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=1, padding=50, dilation=1, padding_mode='circular'),#, padding_mode='zeros'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=1, padding=52, dilation=10, padding_mode='circular'),#, padding_mode='zeros'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.final_temperal_kernel_size, stride=2, padding=0, dilation=1, padding_mode='circular'),#, padding_mode='zeros')
            )
        elif num_input_time_steps == 1001: # case for defalut net with 1000k pred
            self.initial_temporal_kernel_size = 9 # SMALL TEMPORAL CONV LAYER
            self.final_temperal_kernel_size = self.num_input_time_steps//2 - (1-self.num_input_time_steps%2)
            # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
            self.temporal_conv_layers = nn.Sequential( 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=1, padding=4, dilation=1, padding_mode='circular'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.final_temperal_kernel_size, stride=1, padding=(self.final_temperal_kernel_size//2)-1, dilation=1, padding_mode='circular')
            )
        else: # case for defalut net with 100 pred steps / case for nC=25, nK=9, nF=104
            self.initial_temporal_kernel_size = 10 # SMALL TEMPORAL CONV LAYER
            self.final_temperal_kernel_size = (self.num_input_time_steps//10)//2
            # SMALL TEMPORAL CONV LAYER - scales (slowly) with size relative to the input, designed for temporal windows 2000/5000/10000
            self.temporal_conv_layers = nn.Sequential( 
                nn.Conv1d(num_channels, num_channels, self.initial_temporal_kernel_size, stride=self.initial_temporal_kernel_size, padding=self.initial_temporal_kernel_size, dilation=1, padding_mode='circular'), 
                nn.ReLU(), 
                nn.Conv1d(num_channels, num_channels, self.final_temperal_kernel_size, stride=1, padding=2, dilation=1)#, padding_mode='circular')
            )

        self.spacial_conv_layer = nn.Sequential( 
            nn.Conv1d(num_input_time_steps, num_input_time_steps, self.spacial_kernel_size, stride=1, padding=num_channels//2, dilation=1, padding_mode='circular'),#padding=num_channels-1, dilation=1, padding_mode='zeros'), 
            nn.ReLU(), 
        )
        # # LARGE TEMPORAL CONV LAYER - scales with the size of the input, too large for available hardware atm
        # self.temporal_conv_layers = nn.Sequential( 
        #     nn.Conv1d(num_channels, num_channels, num_input_time_steps//2, stride=1, padding=0, dilation=1),#, padding_mode='zeros'), 
        #     nn.ReLU(), 
        #     nn.Conv1d(num_channels, latent_node_state_dim, (num_input_time_steps//2)-(latent_node_state_dim-2), stride=1, padding=0, dilation=1),#, padding_mode='zeros')
        # )
        
        pass
    
    def forward(self, x):
        # print("BasicConv1dEncoder.forward: 1st x.size() == ", x.size())
        x = torch.transpose(x, 1, 2) # prepare for spacial convolution
        # print("BasicConv1dEncoder.forward: 2nd x.size() == ", x.size())
        x = self.spacial_conv_layer(x)
        # print("BasicConv1dEncoder.forward: 3rd x.size() == ", x.size())
        x = torch.transpose(x, 1, 2) # swap channel and temporal dim in preparation for temporal convolutions
        # print("BasicConv1dEncoder.forward: 4th x.size() == ", x.size())
        x = self.temporal_conv_layers(x)
        # print("BasicConv1dEncoder.forward: 5th x.size() == ", x.size())
        return x.view(x.size()[0], self.num_channels, -1)#self.latent_node_state_dim)


class BasicDiscriminatorNet(torch.nn.Module):
    def __init__(self, num_channels, temp_dim, out_dim=1):
        super(BasicDiscriminatorNet, self).__init__()
        self.in_dim = num_channels*temp_dim
        self.num_channels = num_channels
        self.temp_dim = temp_dim
        self.out_dim = out_dim

        self.kernel_1_size = 3
        self.conv1_out_chans = num_channels*3
        self.kernel_Final_size = 5
        self.convFinal_out_chans = 10
        if temp_dim == 1201:
            print("BasicDiscriminatorNet.__init__: ADDING EXTRA PADDING TO 1ST CONV")
            self.padding_1_amount = 20+(self.temp_dim % self.kernel_1_size) // 2
            self.padding_2_amount = 5
            self.post_conv_dim = 28*self.convFinal_out_chans
        else:
            self.padding_1_amount = (self.temp_dim % self.kernel_1_size) // 2
            self.padding_2_amount = 1
            self.post_conv_dim = 45*self.convFinal_out_chans
        print("BasicDiscriminatorNet.__init__: self.padding_1_amount == ", self.padding_1_amount)
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self.num_channels, self.conv1_out_chans, self.kernel_1_size, stride=self.kernel_1_size, padding=self.padding_1_amount), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.conv1_out_chans, self.conv1_out_chans, self.kernel_1_size, stride=self.kernel_1_size, padding=self.padding_2_amount), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(self.conv1_out_chans, self.convFinal_out_chans, self.kernel_Final_size, stride=self.kernel_Final_size, padding=1), 
            torch.nn.ReLU(),
        )

        print("BasicDiscriminatorNet.__init__: ALLOCATING FINAL-PRED LINEAR BLOCK")
        self.num_final_preds_to_process = min(100, self.temp_dim)
        self.focus_hidden_dim = 800
        self.focussed_linear = torch.nn.Sequential(
            torch.nn.Linear(self.num_channels*self.num_final_preds_to_process, self.focus_hidden_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.focus_hidden_dim, self.focus_hidden_dim)
        )

        self.combined_embed_dim = self.post_conv_dim+self.focus_hidden_dim
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(self.combined_embed_dim, self.combined_embed_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(self.combined_embed_dim, self.out_dim), 
            torch.nn.Sigmoid()
        )
        pass

    def forward(self, x):
        # print("BasicDiscriminatorNet.forward: orig x.size() == ", x.size())
        x_tail = torch.flatten(x[:,:,-1*self.num_final_preds_to_process:], start_dim=1)
        # print("BasicDiscriminatorNet.forward: pre-focussed_linear x_tail.size() == ", x_tail.size())
        tail_embed = self.focussed_linear(x_tail).view(-1, self.focus_hidden_dim)
        # print("BasicDiscriminatorNet.forward: tail_embed.size() == ", tail_embed.size())
        # print("BasicDiscriminatorNet.forward: pre-conv x.size() == ", x.size())
        x = self.conv_layers(x)
        # print("BasicDiscriminatorNet.forward: post-conv x.size() == ", x.size())
        x = torch.flatten(x, start_dim=1).view(-1, self.post_conv_dim)
        # print("BasicDiscriminatorNet.forward: post-flatten x.size() == ", x.size())
        x = torch.cat([x, tail_embed], dim=1).view(-1, self.combined_embed_dim)
        # print("BasicDiscriminatorNet.forward: post-concat x.size() == ", x.size())
        return self.linear_layers(x).view(-1, self.out_dim)
# ===================================================================================================


# Loss Functions ===============================================================================================================
class VanillaGANLoss(torch.nn.Module):
    def __init__(self, disc_coeff):
        """
        Discriminator Loss Computation
        """
        super(VanillaGANLoss, self).__init__()
        self.disc_coeff = disc_coeff
        self.bce_loss = torch.nn.BCELoss()
        pass

    def forward(self, p_x_hat, true_x_disc_label, mu, sigma): # has same api as VariationalGANLoss for rapid prototyping
        reconstruction_loss = -1*self.bce_loss(p_x_hat, true_x_disc_label)
        combined_loss = self.disc_coeff*reconstruction_loss
        return combined_loss, (reconstruction_loss.detach().item(), None)


class GANLossWithMSE(torch.nn.Module):
    def __init__(self, disc_coeff=100, mse_coeff=10):
        super(GANLossWithMSE, self).__init__()
        self.disc_coeff = disc_coeff
        self.mse_coeff = mse_coeff
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        pass
    
    def forward(self, gen_preds, gen_targets, disc_preds, disc_targets):
        # print("GANLossWithMSE.forward: disc_preds.size() == ", disc_preds.size())
        # print("GANLossWithMSE.forward: disc_targets.size() == ", disc_targets.size())
        disc_loss = -1*self.bce_loss(disc_preds, disc_targets)
        # print("GANLossWithMSE.forward: gen_preds.size() == ", gen_preds.size())
        # print("GANLossWithMSE.forward: gen_targets.size() == ", gen_targets.size())
        mse_loss = self.mse_loss(gen_preds, gen_targets)
        combined_loss = self.disc_coeff*disc_loss + self.mse_coeff*mse_loss
        return combined_loss, (disc_loss.detach().item(), mse_loss.detach().item())
        
class GeneratorLossWithMSE(torch.nn.Module):
    def __init__(self, mse_coeff=10):
        super(GeneratorLossWithMSE, self).__init__()
        self.mse_coeff = mse_coeff
        self.mse_loss = torch.nn.MSELoss()
        pass
    
    def forward(self, gen_preds, gen_targets):
        return self.mse_coeff*self.mse_loss(gen_preds, gen_targets)

class DiscriminatorLossWithBCE(torch.nn.Module):
    def __init__(self, disc_coeff=100):
        super(DiscriminatorLossWithBCE, self).__init__()
        self.disc_coeff = disc_coeff
        self.bce_loss = torch.nn.BCELoss()
        pass
    
    def forward(self, disc_preds, disc_targets):
        return self.disc_coeff*self.bce_loss(disc_preds, disc_targets) 
# ==============================================================================================================================


# Other Utils ==================================================================================================================

# ==============================================================================================================================
