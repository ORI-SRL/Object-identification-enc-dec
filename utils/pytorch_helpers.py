import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
import random


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def learn_model(model, train_loader, test_loader, optimizer, criterion, n_epochs=50, max_patience=10, save_folder='./',
                save=True, show=True):
    model_name = model.__class__.__name__
    print(model_name)

    device = get_device()
    print(device)
    model.to(device)

    # Epochs
    train_loss_out = []
    test_loss_out = []

    patience = 0
    best_loss = None
    best_params = None
    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        test_loss = 0.0

        # Training
        model.train()
        for data in train_loader:
            frame = data["data"].to(device)
            # labels = data["labels"]

            optimizer.zero_grad()
            outputs, embeddings = model(frame)
            loss = criterion(outputs, frame)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_loss_out.append(train_loss)

        model.eval()
        for data in test_loader:
            frame = data["data"].to(device)

            outputs, embeddings = model(frame)
            loss2 = criterion(outputs, frame)
            loss2.backward()
            test_loss += loss2.item()

        test_loss = test_loss / len(test_loader)
        test_loss_out.append(test_loss)

        # EARLY STOPPING LOGIC
        if best_loss is None or test_loss < best_loss:
            best_loss = test_loss
            patience = 0
            best_params = copy.copy(model.state_dict())
        else:
            patience += 1
            if patience >= max_patience:
                print(f'Early stopping: training terminated at epoch {epoch} due to es, '
                      f'patience exceeded at {max_patience}')
                break

        # luca: we observe loss*1e3 just for convenience. the loss scaling isn't necessary above
        print('Epoch: {} \tTraining Loss: {:.8f} \tTesting loss: {:.8f}'.format(epoch, train_loss*1e3, test_loss*1e3))

    if save and best_params is not None:
        model_file = f'{save_folder}{model_name}_model_state.pt'
        torch.save(best_params, model_file)

    if show:
        # plot model losses
        x = list(range(1, len(test_loss_out) + 1))
        plt.plot(x, train_loss_out, label=model_name+"Training loss")
        plt.plot(x, test_loss_out, label=model_name+"Testing loss")
        plt.xlabel('epoch #')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return best_params, test_loss_out


def test_model(model, train_loader, test_loader, save_folder='./', save=True, show=True):
    model_name = model.__class__.__name__
    print(model_name)

    device = get_device()
    print(device)
    model.to(device)

    model.eval()
    for data in test_loader:
        frame = data["data"].to(device)
        outputs, embeddings = model(frame)


    if save:
        model_file = f'{save_folder}{model_name}_model_state.pt'
        torch.save(embeddings, model_file)

    if show:
        # plot encoded data
        torch.Tensor.ndim = property(lambda self: len(self.shape))
        x = embeddings[:, 0].cpu()
        y = embeddings[:, 1].cpu()
        x = x.detach().numpy()
        y = y.detach().numpy()
        plt.scatter(x, y, label=model_name+"_encoded_data")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()
