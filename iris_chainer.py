import chainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
import numpy as np

x, t = load_iris(return_X_y=True)

x = x.astype('float32')
t = t.astype('int32')

x_train_val, x_test, t_train_val, t_test  = train_test_split(x, t, test_size = 0.3, random_state = 0)

x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)

l = L.Linear(3, 2)

n_input = 4
n_hidden = 6
n_output = 3

net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

optimizer = chainer.optimizers.SGD(lr = 0.05)

optimizer.setup(net)

n_epoch = 30
n_batchsize = 16
iteration = 0

results_train={
    'loss' : [],
    'accuracy' : []
}

results_valid={
    'loss' : [],
    'accuracy' : []
}

for epoch in range(n_epoch):
    order = np.random.permutation(len(x_train))
    
    loss_list = []
    accuracy_list = []
    
    for i in range(0, len(order), n_batchsize):
        index = order[i:i+n_batchsize]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]
        
        y_train_batch = net(x_train_batch)
        
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)
        
        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)
        
        net.cleargrads()
        loss_train_batch.backward()
        
        optimizer.update()
        
        iteration += 1
        
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)
    
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_val)

    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.softmax_cross_entropy(y_val, t_val)

    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'\
        .format(epoch, iteration, loss_train, loss_val.array))

    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)