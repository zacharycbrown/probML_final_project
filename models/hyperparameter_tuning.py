import time
import torch
from ax.service.ax_client import AxClient
# from ax.service.utils.instantiation import ObjectiveProperties
import gc
import os

from models.simple_gan import SimpleGAN
from models.simple_2d_gan import Simple2DGAN
from models.nonlinear_factor_model_2d_gan import NFM2DGAN
from models.utils import GANLossWithMSE, GeneratorLossWithMSE, DiscriminatorLossWithBCE


def create_Ax_client(exp_name, exp_params, exp_objective_name, minimize=True):
    ax_client = AxClient()
    ax_client.create_experiment(
        name=exp_name, # e.g. "basic_demo"
        parameters=exp_params, # e.g. exp_params=[{"name": "x", "type": "range", "bounds": [-100., 100.], "value_type": "float"},],
        objective_name=exp_objective_name, # e.g. "f"
        minimize=minimize
    )
    return ax_client

def evaluate_training_params(parameters, model_type="NFM2DGAN", criteria="VAL_MSE"):
    model = None
    results = None
    
    # load model
    torch.cuda.empty_cache()
    model = None
    if model_type == "NFM2DGAN":
        model = NFM2DGAN(parameters)
    else:
        raise NotImplementedError()
    model = model.to(model.device)

    # evaluate model
    if criteria == "VAL_MSE":
        _, _, _, _, _, avg_val_gen_MSE_losses, _, _ = model.fit(parameters)
        results = avg_val_gen_MSE_losses[-1]
    elif criteria == "VAL_INV_DISC":
        _, _, _, _, avg_val_gen_disc_losses, _, _, _ = model.fit(parameters)
        results = avg_val_gen_disc_losses[-1]
    else:
        raise ValueError("evaluate_training_params: unrecognized value of criteria == "+str(criteria))

    # close out and return results
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


def run_grid_search(exp_name, exp_params, complex_params, exp_objective_name, log_save_dir, minimize=True, num_trials=50, model_type="NMF2DGAN", criteria="VAL_MSE"):
    ax_client = create_Ax_client(exp_name, exp_params, exp_objective_name, minimize=minimize)
    param_points = []
    param_vals = []
    
    # iterate over experiments: see https://mengliuz.medium.com/hyperparameter-tuning-for-deep-learning-models-with-the-ax-57c6f117a31b
    print("run_grid_search: BEGINNING EXPERIMENTS")
    counter = 0
    all_trial_counter = 0
    while counter < num_trials:
        print("successful trial counter==", counter)
        print("all_trial_counter==", all_trial_counter)
        start = time.time()
        params, trial_ind = ax_client.get_next_trial()
        # print("params == ", params)
        for cKey in complex_params.keys():
            params[cKey] = complex_params[cKey]
        # print("post complex-insert params == ", params)

        gen_burnin_loss_fn = GeneratorLossWithMSE(params["mse_coeff"])
        gen_loss_fn = GANLossWithMSE(params["disc_coeff"], params["mse_coeff"])
        disc_loss_fn = DiscriminatorLossWithBCE(params["disc_coeff"])
        params["gen_burnin_loss_fn"] = gen_burnin_loss_fn
        params["gen_loss_fn"] = gen_loss_fn
        params["disc_loss_fn"] = disc_loss_fn

        param_points.append(params)

        try: # FOR DEBUGGING
            curr_result = evaluate_training_params(params, model_type, criteria)
            counter += 1
            print("<<< SUCCESSFUL EXPERIMENT ", counter, " COMPLETED! >>>")
            file = open(log_save_dir+os.sep+"run_grid_search_logging.txt", 'a') # see https://www.scaler.com/topics/append-to-file-python/
            file.write("\tSuccessful run "+str(counter)+": params=="+str(params)+" result=="+str(curr_result))
            file.close()
        except Exception as e: # FOR DEBUGGING
            print("-----")
            print("Experiment yielded exception e == ", e)
            print("-----")
            curr_result = 1e20

        param_vals.append(curr_result)
        ax_client.complete_trial(trial_index=trial_ind, raw_data={"f":curr_result})
        runtime = time.time() - start # see https://pynative.com/python-get-execution-time-of-program/
        print("\trun-time: ", runtime)
        
        all_trial_counter += 1

        # if counter==1: # FOR DEBUGGING
        #     print("run_grid_search: BREAKING FOR DEBUGGING PURPOSES") # FOR DEBUGGING
        #     break # FOR DEBUGGING

    best_params, metrics = ax_client.get_best_parameters()
    print("run_grid_search: best_params == ", best_params)
    return best_params, metrics, param_points, param_vals

