import matplotlib.pyplot as plt

def plot_train(train_losses, test_losses, train_r2s, test_r2s, dataset, data_type, model_type):
    """Plots the training and test losses and R2s

    Args:
        train_losses (list): list of lists training losses
        test_losses (list): list of lists of test losses
        train_r2s (list): list of lists of training R2s
        test_r2s (list): list of lists of test R2s

    Returns:
        None
    """
    figure, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,5))
    plt.figure(figsize=(20,5))
    ax1.set_title("Train and Test Loss", fontsize=22)
    for i in range(len(train_losses)):
        ax1.plot(train_losses[i],label="train_{}".format(i))
        ax1.plot(test_losses[i],label="test_{}".format(i))
    ax1.set_xlabel("epoch", fontsize=18)
    ax1.set_ylabel("loss", fontsize=18)
    ax1.legend()
    ax1.set_ylim([min(map(min, train_losses)),max(map(max, train_losses))*1.1])

    ax2.set_title("Train and Test R-squared", fontsize=22)
    for i in range(len(train_r2s)):
        ax2.plot(train_r2s[i],label="train_{}".format(i))
        ax2.plot(test_r2s[i],label="test_{}".format(i))
    ax2.set_xlabel("epoch", fontsize=18)
    ax2.set_ylabel("r2", fontsize=18)
    #ax2.set_ylim([min(map(min, train_r2s)),max(map(max, train_r2s))*1.1])
    ax2.set_ylim(-1,1)
    ax2.legend()
    path = 'plots/{}_{}_{}.png'.format(dataset, data_type, model_type)
    figure.savefig(path)


     
