import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import pickle


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            #print(feature.shape)
            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1

            corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * corrects/batch.batch_size


            if steps % args.log_interval == 0:

                # if steps % args.test_interval == 0:
                #     print(type(loss))
                #     print(type(loss.data))
                train_loss.append(float(loss.data[0]))
                train_acc.append(float(accuracy))
                sys.stdout.write(
                    '\rEpoch {} Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(epoch,steps,
                                                                             loss.data[0], 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))



            if steps % args.test_interval == 0:
                dev_acc, dev_loss, dev_corrects, dev_size = eval(dev_iter, model, args)
                val_acc.append(float(dev_acc))
                val_loss.append(float(dev_loss))
                print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(dev_loss,
                                                                   dev_acc,
                                                                   dev_corrects,
                                                                   dev_size))

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
        print(len(train_loss), len(train_acc))
        print(len(val_loss), len(val_acc))
        #break
    # with open('results/train_loss.pkl', 'wb') as f:
    #     pickle.dump(train_loss, f)
    with open('results/256_train_acc.pkl', 'wb') as f:
        pickle.dump(train_acc, f)
    # with open('results/val_loss.pkl', 'wb') as f:
    #     pickle.dump(val_loss, f)
    with open('results/256_val_acc.pkl', 'wb') as f:
        pickle.dump(val_acc, f)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    # print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
    #                                                                    accuracy,
    #                                                                    corrects,
    #                                                                    size))
    return accuracy, avg_loss, corrects, size


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
