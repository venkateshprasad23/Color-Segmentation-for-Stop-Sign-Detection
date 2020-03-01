#!/usr/bin/python

from scipy.special import expit
import numpy as np
import os
import sklearn.metrics as skmetric

def safe_log(z, minval=0.0000000001):
    return np.log(z.clip(min=minval))

def loss_calculate(X, y, w):
    z = np.matmul(X, w)
    t1 = y * safe_log(expit(z))
    t2 = (1 - y) * safe_log(1 - expit(z))
    t = t1 + t2
    ret = t.sum()
    return -1 * ret


def grad_log_likelihood(X, y, w):
    sig = expit(np.matmul(X, w))
    coef = y - sig
    gradua = coef * X
    susu = np.sum(gradua, axis=0)
    ret = np.reshape(susu, w.shape)
    return ret

def reload_utility(w, l, e):
    save_dir = "./savedoutput/training/"
    wei_name = "weights_" + str(e)
    l_name = "loss_" + str(e)
    np.save(save_dir + wei_name + ".npy", w)
    np.save(save_dir + l_name + ".npy", l)
    np.savetxt(save_dir + wei_name + ".txt", w)

    with open(save_dir + l_name + ".txt", "w") as f:
        f.write(str(l))


def metrics(l, a, f):
    save_dir = "./savedoutput/metrics/"
    loss_name = "val_loss"
    accuracy_name = "val_acc"
    f_name = "val_f1"
    np.save(save_dir + loss_name + ".npy", l)
    np.save(save_dir + accuracy_name + ".npy", a)
    np.save(save_dir + f_name + ".npy", f)
    with open(save_dir + loss_name + ".txt", "w") as fi:
        fi.write(str(l))
    with open(save_dir + accuracy_name + ".txt", "w") as fi:
        fi.write(str(a))
    with open(save_dir + f_name + ".txt", "w") as fi:
        fi.write(str(f))


def main():

    train_img_dir = "./data/train/images/"
    train_mask_dir = "./data/train/segmented/"


    val_img_dir = "./data/val/images/"
    val_mask_dir = "./data/val/segmented/"


    samp = np.load(train_img_dir + "3.npy")
    num_of_channel = samp.shape[2]


    w = np.random.randn(num_of_channel + 1, 1)
    alpha = 0.01
    epochs = 50



    sum_loss = 0
    avg_loss = 0
    step = 0

    for e in range(epochs):
        print("In epoch number:", e + 1)
        for file_name in os.listdir(train_img_dir):
            X = np.load(os.path.join(train_img_dir, file_name))
            X = np.reshape(X, (X.shape[0] * X.shape[1], 3))
            one_row = np.ones((X.shape[0], 1))
            X = np.concatenate((one_row, X), 1)
            y = np.load(os.path.join(train_mask_dir, file_name))
            y = np.reshape(y, (y.shape[0] * y.shape[1], 1))

            l = loss_calculate(X, y, w)
            print("Loss : ", l)
            sum_loss += l
            step += 1
            avg_loss = sum_loss / step


            w = w + alpha * grad_log_likelihood(X, y, w)

        print("Average loss after Epoch: {} is {}".format(e + 1, avg_loss))
        # save the average loss, and the weights after every epoch of training
        reload_utility(w, avg_loss, e + 1)

    avg_val_loss = 0
    avg_val_acc = 0
    avg_f1_sc = 0
    step = 0
    su_loss = 0
    su_acc = 0
    su_f1 = 0
    # Validation metrics
    for file_name in os.listdir(val_img_dir):
        X_val = np.load(os.path.join(val_img_dir, file_name))
        X_val = np.reshape(X_val, (X_val.shape[0] * X_val.shape[1], 3))
        o = np.ones((X_val.shape[0], 1))
        X_val = np.concatenate((o, X_val), 1)
        y_val = np.load(os.path.join(val_mask_dir, file_name))
        y_val = np.reshape(y_val, (y_val.shape[0] * y_val.shape[1], 1))

        y_pred = np.matmul(X_val, w) >= 0
        y_pred = y_pred.astype(np.uint8)

        # Metrics calculated here
        l = loss_calculate(X_val, y_val, w)
        step += 1
        su_loss += l
        avg_val_loss = su_loss / step
        acc = skmetric.accuracy_score(y_val, y_pred)
        su_acc += acc
        avg_val_acc = su_acc / step
        f1 = skmetric.f1_score(y_val, y_pred)
        su_f1 += f1
        avg_f1_sc = su_f1 / step

        # Print computed metrics here
        print("Loss:", l)
        print("Accuracy:", acc)
        print("F1 score: ", f1)

    # save the validation metrics over all the validation images to disk
    metrics(avg_val_loss, avg_val_acc, avg_f1_sc)


if __name__ == "__main__":
    main()
