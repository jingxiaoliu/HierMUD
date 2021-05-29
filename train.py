import numpy as np
import torch
import os
from tqdm import tqdm_notebook as tqdm
import argparse
from torch.autograd import Variable

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer

#==============eval
def evaluate(model_instance,input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]                
        lab_tmp = data[1].clone()
        lab_tmp = torch.where(torch.remainder(data[1],2).byte(),lab_tmp,(data[1]/2).long())
        lab_tmp = torch.where(torch.remainder(data[1]+1,2).byte(),lab_tmp,((data[1]-1)/2).long())
        labels = lab_tmp
        labels_2 = torch.remainder(data[1],2)
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels_2 = Variable(labels_2.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            labels_2 = Variable(labels_2)
        softmax_outputs = model_instance.predict(inputs)
        probabilities = softmax_outputs[0]
        probabilities_2 = softmax_outputs[1]

        probabilities = torch.exp(probabilities.data.float())
        if probabilities_2 is not None:
            probabilities_2 = torch.exp(probabilities_2.data.float())
        labels = labels.data.float()
        labels_2 = labels_2.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            if probabilities_2 is not None:                
                all_probs_2 = probabilities_2
                all_labels_2 = labels_2
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            if probabilities_2 is not None:       
                all_probs_2 = torch.cat((all_probs_2, probabilities_2), 0)
                all_labels_2 = torch.cat((all_labels_2, labels_2), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels) / float(all_labels.size()[0])
    accuracy_ = torch.sum(torch.remainder(torch.squeeze(predict).float(),2) == all_labels_2) / float(all_labels_2.size()[0])

    if probabilities_2 is None:
        all_probs_2 = None
        accuracy_2 = None
    else:
        _, predict_2 = torch.max(all_probs_2, 1)
        accuracy_2 = torch.sum(torch.squeeze(predict_2).float() == all_labels_2) / float(all_labels_2.size()[0])
        all_probs_2 = all_probs_2.cpu().detach().numpy()
        accuracy_2 = accuracy_2.cpu().detach().numpy()

    model_instance.set_train(ori_train_state)
    return accuracy.cpu().detach().numpy(),all_probs.cpu().detach().numpy(),accuracy_2,all_probs_2,accuracy_.cpu().detach().numpy()

def get_bottleneck(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]           
        lab_tmp = data[1].clone()
        lab_tmp = torch.where(torch.remainder(data[1],2).byte(),lab_tmp,(data[1]/2).long())
        lab_tmp = torch.where(torch.remainder(data[1]+1,2).byte(),lab_tmp,((data[1]-1)/2).long())
        labels = lab_tmp
        labels_2 = torch.remainder(data[1],2)
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            labels_2 = Variable(labels_2.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
            labels_2 = Variable(labels_2)
        features, _,_,_ = model_instance.embedding(inputs)

        feature1 = features[0].data.float()
        feature2 = features[1].data.float()
        if len(features)==3:
          feature3 = features[2].data.float()
        labels = labels.data.float()
        labels_2 = labels_2.data.float()
        if first_test:
            all_feas1 = feature1
            all_feas2 = feature2
            if len(features)==3:
              all_feas3 = features3
            else:
              all_fea3 = None
            all_labels = labels
            all_labels_2 = labels_2
            first_test = False
        else:
            all_feas1 = torch.cat((all_feas1, feature1), 0)
            all_feas2 = torch.cat((all_feas2, feature2), 0)
            if len(features)==3:
              all_feas3 = torch.cat((all_feas3, feature3), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            all_labels_2 = torch.cat((all_labels_2, labels_2), 0)

    model_instance.set_train(ori_train_state)
    return all_feas1.cpu().detach().numpy(),all_feas2.cpu().detach().numpy(),\
        all_feas3.cup().detach().numpy(),\
        all_labels.cpu().detach().numpy(),all_labels_2.cpu().detach().numpy()

def train(model_instance, train_source_loader, train_target_loader, test_target_loader,
          group_ratios, max_iter, optimizer, lr_scheduler, eval_interval):
    model_instance.set_train(True)
    print("start train...")
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm(desc='Train iter', total=max_iter)    
    first_itr = True
    target_acc1 = []
    target_acc2_ = []
    target_acc2 = []

    while True:
        for (datas1, datas2, datat1, datat2) in tqdm(
                zip(train_source_loader[0], train_source_loader[1], train_target_loader[0], train_target_loader[1]),
                total=min(len(train_source_loader[0]), len(train_source_loader[1]), len(train_target_loader[0]), len(train_target_loader[1])),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source1 = datas1[0]
            inputs_target1 = datat1[0]   
            inputs_source2 = datas2[0]
            inputs_target2 = datat2[0]   
            labels_source = []
            labels_target = []       
            
            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source1, inputs_target1 = Variable(inputs_source1).cuda(), Variable(inputs_target1).cuda()
                inputs_source2, inputs_target2 = Variable(inputs_source2).cuda(), Variable(inputs_target2).cuda()
                lab_tmp = datas1[1].clone()
                lab_tmp = torch.where(torch.remainder(datas1[1],2).byte(),lab_tmp,(datas1[1]/2).long())
                lab_tmp = torch.where(torch.remainder(datas1[1]+1,2).byte(),lab_tmp,((datas1[1]-1)/2).long())
                labels_source.append(Variable(lab_tmp).cuda())
                lab_tmp = datat1[1].clone()
                lab_tmp = torch.where(torch.remainder(datat1[1],2).byte(),lab_tmp,(datat1[1]/2).long())
                lab_tmp = torch.where(torch.remainder(datat1[1]+1,2).byte(),lab_tmp,((datat1[1]-1)/2).long())
                labels_target.append(Variable(lab_tmp).cuda())   
                labels_source.append(Variable(torch.remainder(datas2[1],2)).cuda())
                labels_target.append(Variable(torch.remainder(datat2[1],2)).cuda())
            else:
                inputs_source1, inputs_target1 = Variable(inputs_source1), Variable(inputs_target1)
                inputs_source2, inputs_target2 = Variable(inputs_source2), Variable(inputs_target2)
                lab_tmp = datas1[1].clone()
                lab_tmp = torch.where(torch.remainder(datas1[1],2).byte(),lab_tmp,(datas1[1]/2).long())
                lab_tmp = torch.where(torch.remainder(datas1[1]+1,2).byte(),lab_tmp,((datas1[1]-1)/2).long())
                labels_source.append(Variable(lab_tmp))
                lab_tmp = datat1[1].clone()
                lab_tmp = torch.where(torch.remainder(datat1[1],2).byte(),lab_tmp,(datat1[1]/2).long())
                lab_tmp = torch.where(torch.remainder(datat1[1]+1,2).byte(),lab_tmp,((datat1[1]-1)/2).long())
                labels_target.append(Variable(lab_tmp))   
                labels_source.append(Variable(torch.remainder(datas2[1],2)))
                labels_target.append(Variable(torch.remainder(datat2[1],2))) 

            inputs_source = [inputs_source1, inputs_source2]
            inputs_target = [inputs_target1, inputs_target2]

            classifier_loss,transfer_loss,oth_diff_s,oth_diff_t = train_batch(model_instance, inputs_source,\
             labels_source, inputs_target, optimizer)

            if first_itr:
                cls_losses = classifier_loss
                tsf_losses = transfer_loss
                oth_diffs_s = oth_diff_s
                oth_diffs_t = oth_diff_t
                first_itr = False
            else:
                cls_losses=np.append(cls_losses,classifier_loss)
                tsf_losses=np.append(tsf_losses,transfer_loss)
                oth_diffs_s=np.append(oth_diffs_s,oth_diff_s)
                oth_diffs_t=np.append(oth_diffs_t,oth_diff_t)

            # val
            if iter_num % eval_interval == 1 and iter_num != 0:
                eval_result_tgt,all_probs_tgt,eval_result_tgt_2,all_probs_tgt_2,eval_acc_t = evaluate(model_instance, test_target_loader[0])
                if iter_num % 2000 == 1:
                  print({'Tgt accuracy':eval_result_tgt})
                  if eval_result_tgt_2 is not None:
                      print({'Tgt accuracy 2':eval_result_tgt_2})
                      print({'Tgt accuracy 2 from 1':eval_acc_t})
                eval_result_src,all_probs_src,eval_result_src_2,all_probs_src_2,eval_acc_s = evaluate(model_instance, train_source_loader[0])
                if iter_num % 2000 == 1:
                  print({'Src accuracy':eval_result_src})
                  if eval_result_src_2 is not None:
                      print({'Src accuracy 2':eval_result_src_2})
                target_acc1.append(eval_result_tgt)
                target_acc2.append(eval_result_tgt_2)
                target_acc2_.append(eval_acc_t)
                
            iter_num += 1
            total_progress_bar.update(1)
        epoch += 1
        if iter_num >= max_iter:
            break
    print('finish train')
    return all_probs_tgt,all_probs_tgt_2,all_probs_src,all_probs_src_2,cls_losses,tsf_losses,oth_diffs_s,oth_diffs_t,target_acc1,target_acc2,target_acc2_

def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer):
    total_loss, classifier_loss, transfer_loss, oth_diff_s, oth_diff_t\
                 = model_instance.get_loss(inputs_source, inputs_target, labels_source)
    total_loss.backward()
    optimizer.step()
    if oth_diff_s is not None:
        oth_diff_s = oth_diff_s.cpu().detach().numpy()
        oth_diff_t = oth_diff_t.cpu().detach().numpy()
    return classifier_loss.cpu().detach().numpy(),transfer_loss.cpu().detach().numpy(),\
            oth_diff_s,oth_diff_t
