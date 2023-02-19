from matplotlib import pyplot as plt


def plot_curve(output, title, x_axis_name, y_axis_name, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(output)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_train_and_val_roc_auc_score_history(train_scores, val_scores, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(train_scores, label="training")
    ax1.plot(val_scores, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average ROC-AUC SCORE")
    ax1.set_title("Average ROC-AUC Score Over Time")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_train_and_val_acc(train_acc, val_acc, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(train_acc, label="training")
    ax1.plot(val_acc, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Prediction Accuracy Over Time")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_train_and_val_loss(train_loss, val_loss, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(train_loss, label="training")
    ax1.plot(val_loss, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title("Average Training Losses")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_signal(x, save_path):
    fig1, ax1 = plt.subplots()
    ax1.plot(x, label="signal")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Signal Visualization")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_simulation_vs_true_lfp(x, pred_y, y, save_path):
    num_inputs = len(x)
    num_preds = len(pred_y)
    plot_domain = [i for i in range(num_inputs + num_preds)]
    fig1, ax1 = plt.subplots()
    ax1.plot(plot_domain[num_inputs:], y, label="true signal")
    ax1.plot(plot_domain[num_inputs:], pred_y, label="predicted signal")
    ax1.plot(plot_domain[:num_inputs], x, label="input signal")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("True Signal vs Predicted Signal")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    
    # plot zoomed version of pred
    fig1, ax1 = plt.subplots()
    ax1.plot(y, marker="+", label="true signal")
    ax1.plot(pred_y, marker="x", label="predicted signal")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("True Signal vs Predicted Signal")
    plt.legend()
    plt.draw()
    fig1.savefig(save_path[:-4]+"_ZOOMED.png")
    plt.close()
    pass

def plot_factor_scores_across_time_steps(gamma_series, save_path):
    print("general_utils.plotting.plot_factor_scores_across_time_steps: gamma_series.shape == ", gamma_series.shape)
    # see https://matplotlib.org/stable/gallery/images_contours_and_fields/matshow.html#sphx-glr-gallery-images-contours-and-fields-matshow-py
    plt.matshow(gamma_series)
    plt.xlabel("Time Step")
    plt.ylabel("Factor Score")
    plt.title("Factor Scores Across Time")
    plt.legend()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass

def plot_dense_adjacency_matrix(adj, save_path):
    raise NotImplementedError("Required functions for fetching trained adjacency matrix from a trained model are not implemented yet")
    pass

def plot_roc_auc_curve(fpr, tpr, roc_auc_score, plot_series_name, save_path):
    # see https://www.codegrepper.com/code-examples/python/roc+curve+pytorch
    fig1, ax1 = plt.subplots()
    ax1.plot(fpr, tpr, "b", label="AUC=" + str(roc_auc_score))
    ax1.plot([0, 1], [0, 1], "r--")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(plot_series_name + ": Receiver Operating Characteristic (Test)")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass