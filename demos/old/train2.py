from MixtureCDFFlow import *
from scipy.stats import norm


m = Normal(torch.tensor([2.0]), torch.tensor([1.0]))

# applies gradient steps for each mini-batch in an epoch
def train(model, optimizer):
    model.eval()
    with torch.no_grad():
        x = model.sample(200)
        
    model.train()
    x = x.to(ptu.device).float()
    
    fx = m.log_prob(x).exp()
    loss = model.kl(x, fx)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_epochs(model, train_args):
    # training parameters
    epochs, lr = train_args['epochs'], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # plotting parameters
    plot = train_args.get('plot', True)
    plot_frequency = train_args.get('plot_frequency', 5)
    if 'epochs_to_plot' in train_args.keys():
        plot_frequency = epochs + 1
        epochs_to_plot = train_args['epochs_to_plot']
    else:
        epochs_to_plot = []

    train_losses, test_losses = [], []
    for epoch in tqdm_notebook(range(epochs), desc='Epoch', leave=False):
        model.train()
        train_loss = train(model, optimizer)
        # train_loss = eval_loss(model)
        train_losses.append(train_loss)

        # if test_loader is not None:
        #     test_loss = eval_loss(model, test_loader)
        #     test_losses.append(test_loss)

        if plot and (epoch % plot_frequency == 0 or epoch in epochs_to_plot):
            model.plot(f'Epoch {epoch}')
            
    if plot:
        plot_train_curves(epochs, train_losses, train_losses, title='Training Curve')
    return train_losses, train_losses


# SEED = 10
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# cdf_flow_model = MixtureCDFFlow(base_dist='uniform', mixture_dist='gaussian',
#                     n_components=5).to(ptu.device)
# cdf_flow_model_old = copy.deepcopy(cdf_flow_model)
# train_epochs(cdf_flow_model,  
#                       dict(epochs=20, lr=5e-3, epochs_to_plot=[0,5,8,11,49]))



