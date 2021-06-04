import torch.nn as nn
import backbone
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,coeff):
        ctx.coeff = coeff
        output = input * 1.0
        return output.view_as(output)
    @staticmethod
    def backward(ctx,grad_output):
        coeff = ctx.coeff
        return coeff * grad_output.neg(), None

class MUD_net(nn.Module):
    def __init__(self, base_net_sha='mynet1', width=256, class_num=13):
        super(MUD_net, self).__init__()
        ## set base network
        self.class_num = class_num
        self.task_num = len(class_num)
        self.classifier_layers_list = []
        self.bottleneck_layer_list = []

        # task-shared feature extractor
        self.base_network_sha = backbone.network_dict[base_net_sha]()
        bottleneck_dim = self.base_network_sha.output_num()
        # hierarchy extractor        
        self.hier_extractor_list = [nn.Linear(bottleneck_dim, bottleneck_dim),\
                                    nn.BatchNorm1d(bottleneck_dim),\
                                    nn.ReLU(),\
                                    nn.Dropout(0.25)]
        self.hier_extractor_layer = nn.Sequential(*self.hier_extractor_list)
        # task-specific feature extractors and classifiers
        for i in range(self.task_num):
            self.classifier_layers_list.append([nn.Linear(bottleneck_dim,\
                                            width),\
                                            nn.BatchNorm1d(width),\
                                            nn.ReLU(),\
                                            nn.Dropout(0.25),\
                                            nn.Linear(width, width),\
                                            nn.BatchNorm1d(width),\
                                            nn.ReLU(),\
                                            nn.Dropout(0.25),\
                                            nn.Linear(width, class_num[i])])
        self.classifier_layer = nn.ModuleList([nn.Sequential(*self.classifier_layers_list[i])\
                                         for i in range(self.task_num)])
        # domain classifier
        self.grl_layer = GradientReverseLayer()


        self.domain_classifier_layer_list = [nn.Linear(bottleneck_dim, width),\
                                    nn.BatchNorm1d(width),\
                                    nn.ReLU(),\
                                    nn.Dropout(0.25),\
                                    nn.Linear(width, 2)]
        self.domain_classifier_layer = nn.Sequential(*self.domain_classifier_layer_list)

        self.domain_classifier_layer2_list = [nn.Linear(bottleneck_dim, width),\
                                    nn.BatchNorm1d(width),\
                                    nn.ReLU(),\
                                    nn.Dropout(0.25),\
                                    nn.Linear(width, 2)]
        self.domain_classifier_layer2 = nn.Sequential(*self.domain_classifier_layer2_list)


        self.softmax = nn.Softmax(dim=1)

        ## collect parameters
        classifier_parameter_list = [{"params":a.parameters(), "lr":1} for a in self.classifier_layer]
        self.parameter_list = classifier_parameter_list+\
                              [{"params":self.base_network_sha.parameters(), "lr":1},
                              {"params":self.hier_extractor_layer.parameters(), "lr":1},
                                {"params":self.domain_classifier_layer.parameters(), "lr":1},
                                {"params":self.domain_classifier_layer2.parameters(), "lr":1}]

    def forward(self, inputs, coeff):
        outputs = []
        softmax_outputs = []
        shared_fea = []
        outputs_advs = [] 

        shared_fea.append(self.base_network_sha(inputs[0]))
        hier_fea = self.hier_extractor_layer(shared_fea[0])
        outputs.append(self.classifier_layer[0](hier_fea))
        softmax_outputs.append(self.softmax(outputs[0]))
        features_adv = self.grl_layer.apply(shared_fea[0],coeff)
        outputs_advs.append(self.domain_classifier_layer(features_adv))

        shared_fea.append(self.base_network_sha(inputs[1]))
        outputs.append(self.classifier_layer[1](shared_fea[1]))
        softmax_outputs.append(self.softmax(outputs[1]))
        features_adv = self.grl_layer.apply(shared_fea[1],coeff)
        outputs_advs.append(self.domain_classifier_layer(features_adv))
        features_adv = self.grl_layer.apply(hier_fea,coeff)
        outputs_advs.append(self.domain_classifier_layer2(features_adv))
        shared_fea.append(hier_fea)

        return shared_fea, outputs, softmax_outputs, outputs_advs
         
class MUD(object):
    def __init__(self, base_net_sha='mynet1', width=1024, class_num=31, \
        use_gpu=True,max_iter=1000.0,alpha=1.,low_value=0.,high_value=0.5,\
        gamma=10., weight=2.):
        self.c_net = MUD_net(base_net_sha, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        self.task_num = len(class_num)
        self.gamma = gamma
        self.weight = weight
        if self.use_gpu:
            self.c_net = self.c_net.cuda()

        self.max_iter = max_iter
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value

    def get_loss(self, inputs_s, inputs_t, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        domain_labels = torch.cat((torch.zeros(inputs_s[0].size(0)),\
                                    torch.ones(inputs_t[0].size(0)))).long()
        if self.use_gpu:
            domain_labels = domain_labels.cuda()

        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + \
                        np.exp(-self.alpha * self.iter_num / self.max_iter))\
                        - (self.high_value - self.low_value) + self.low_value)

        features_s, outputs_s, _, outputs_adv_s = self.c_net(inputs_s,self.coeff)
        features_t, outputs_t, _, outputs_adv_t = self.c_net(inputs_t,self.coeff)

        classifier_loss1 = class_criterion(outputs_s[0], labels_source[0])
        classifier_loss2 = class_criterion(outputs_s[1], labels_source[1])
        classifier_loss = classifier_loss1+classifier_loss2

        transfer_loss1 = class_criterion(torch.cat((outputs_adv_s[0],outputs_adv_t[0]),0), domain_labels)
        transfer_loss2 = class_criterion(torch.cat((outputs_adv_s[1],outputs_adv_t[1]),0), domain_labels)
        transfer_loss3 = class_criterion(torch.cat((outputs_adv_s[2],outputs_adv_t[2]),0), domain_labels)

        transfer_losses = torch.vstack((transfer_loss1,transfer_loss2))
        transfer_loss = torch.mul(torch.div(torch.log(torch.sum(torch.exp(torch.mul(transfer_losses,\
                    self.gamma)))),self.gamma),self.weight)+transfer_loss3

        self.iter_num += 1

        total_loss = classifier_loss + transfer_loss

        return total_loss, classifier_loss, transfer_loss, None, None

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net([inputs,inputs],self.coeff)
        return softmax_outputs

    def embedding(self, inputs):
        features,outputs,softmax_outputs,outputs_adv = self.c_net([inputs,inputs],self.coeff)
        return features,outputs,softmax_outputs,outputs_adv

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
