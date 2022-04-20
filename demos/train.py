from MixtureCDFFlow import *
# applies gradient steps for each mini-batch in an epoch
def train(model, train_loader, optimizer):
    model.train()
    for x in train_loader:
        x = x.to(ptu.device).float()
        loss = model.nll(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(ptu.device).float()
            loss = model.nll(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss.item()

def train_epochs(model, train_loader, test_loader, train_args):
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
        train(model, train_loader, optimizer)
        train_loss = eval_loss(model, train_loader)
        train_losses.append(train_loss)

        if test_loader is not None:
            test_loss = eval_loss(model, test_loader)
            test_losses.append(test_loss)

        if plot and (epoch % plot_frequency == 0 or epoch in epochs_to_plot):
            model.plot(f'Epoch {epoch}')
            
    if plot:
        plot_train_curves(epochs, train_losses, test_losses, title='Training Curve')
    return train_losses, test_losses


SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

n_train, n_test = 2000, 1000
loader_args = dict(batch_size=128, shuffle=True)
train_loader, test_loader = load_flow_demo_1(n_train, n_test, 
            loader_args, visualize=True, train_only=False)


cdf_flow_model = MixtureCDFFlow(base_dist='uniform', mixture_dist='gaussian',
                    n_components=5).to(ptu.device)
cdf_flow_model_old = copy.deepcopy(cdf_flow_model)
train_epochs(cdf_flow_model, train_loader, test_loader, 
                      dict(epochs=50, lr=5e-3, epochs_to_plot=[0,5,8,11,49]))
visualize_demo1_flow(train_loader, cdf_flow_model_old, cdf_flow_model)


# ud = Uniform(ptu.FloatTensor([0.0]), ptu.FloatTensor([1.0]))
# z = ud.rsample(torch.Size([500]))
# x = cdf_flow_model.invert(z)
# plt.hist(ptu.get_numpy(x), bins=50)



