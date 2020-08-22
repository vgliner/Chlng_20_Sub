import torch
import numpy as np
import pandas as pd
import evaluate_12ECG_score
import matplotlib.pyplot as plt


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss


def save_model(model_to_save, best_acc):
    checkpoints_str = r'Final_fully_connected_net_model.pt'
    saved_state = dict(best_acc=best_acc, ewi=20, model_state=model_to_save.state_dict())
    torch.save(saved_state, checkpoints_str)


def my_loss2(output, target):
    #  G=TP/(TP+FP+2*FN)

    # output.masked_fill_(output>0.0,1.0)    
    # output.masked_fill_(output<=0.0,0.0)    
    TP = torch.mul(output, target)
    TP_mean = torch.mean(TP)
    FP = torch.mul(output, 1 - target)
    FP_mean = torch.mean(FP)
    FN = torch.mul(1 - output, target)
    FN_mean = torch.mean(FN)
    G = TP_mean / (TP_mean + FP_mean + 2 * FN_mean)
    loss = 1 - G
    # loss = torch.mean((output - target)**2)
    return loss


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 9, 25, 9

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

##% Upload data
Data = pd.read_csv('Results_Summary.csv', delimiter=',', header=None).values
x = torch.from_numpy(Data[:, 9:18]).float()
y = torch.from_numpy(Data[:, 0:9]).float()
x_test = Data[:, 9:18]
y_test = Data[:, 0:9]
print(f'X is size of {np.shape(x)} and Y is size of {np.shape(y)}')
print(f'X max is  {x.max()} and Y max is {y.max()}')
print(f'Indexes of nans of X: {np.argwhere(np.isnan(x_test))}')
print(f'Indexes of nans of Y: {np.argwhere(np.isnan(y_test))}')

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.


loss_fn = torch.nn.MSELoss(reduction='mean')
loss_fn = torch.nn.HingeEmbeddingLoss(reduction='mean')

# loss_fn = my_loss2
# loss_fn=evaluate_12ECG_score.G2_loss
learning_rate = 2e-3
logger = []
model_backup = []
best_loss = 1e6
for t in range(50000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)

    if t % 100 == 99:
        if (loss < best_loss) and loss > 0:
            best_loss = loss
            model_backup = model
        print(f'{t}, {loss.item()}, Best loss so far: {best_loss}')
        logger.extend([loss.item()])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

plt.plot(logger)
plt.show()
save_model(model, best_loss)
print('Finished')
