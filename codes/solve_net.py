from utils import LOG_INFO
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[start_idx: end_idx], y[start_idx: end_idx]


def solve_net(model, train_x, train_y, test_x, test_y,
              batch_size, max_epoch, disp_freq, test_freq):

    iter_counter = 0
    loss_list = []
    accuracy_list = []
    test_acc = []
    test_loss = []

    for k in range(max_epoch):
        for x, y in data_iterator(train_x, train_y, batch_size):
            iter_counter += 1
            loss, accuracy = model.train(x, y)
            loss_list.append(loss)
            accuracy_list.append(accuracy)

            if iter_counter % disp_freq == 0:
                msg = 'Training iter %d, mean loss %.5f (batch loss %.5f), mean acc %.5f' % (iter_counter,
                                                                                             np.mean(loss_list),
                                                                                             loss_list[-1],
                                                                                             np.mean(accuracy_list))
                LOG_INFO(msg)
                loss_list = []
                accuracy_list = []

            if iter_counter % test_freq == 0:
                LOG_INFO('    Testing...')
                for tx, ty in data_iterator(test_x, test_y, batch_size, shuffle=False):
                    t_accuracy, t_loss = model.test(tx, ty)
                    test_acc.append(t_accuracy)
                    test_loss.append(t_loss)

                msg = '    Testing iter %d, mean loss %.5f, mean acc %.5f' % (iter_counter,
                                                                              np.mean(test_loss),
                                                                              np.mean(test_acc))
                LOG_INFO(msg)
                test_acc = []
                test_loss = []
