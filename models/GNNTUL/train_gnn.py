import gc
import random
import argparse
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from config import *
from model import *
from dataReader import *
import time

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# 固定随机种子，设置device
device = torch.device('cuda:0' if args.cuda else 'cpu')

# 加载模型，设置优化器
# gnn = BiLSTM_GNN(n_input, n_hidden, n_classes, n_poi, gnn_tag=False, dropout=0.5)
# optimizer = optim.Adam(gnn.parameters(),
#                        lr=learning_rate)
if args.cuda:
    gnn.cuda()

# ------------全局变量-----------------------------------------------------------------------------------------------------------------------------------------

Test_10 = []
result_output = data_input.split('.csv')[0]+'_result.txt'

# ------------训练函数------------------------------------------------------------------------------------------------------------------------------------------


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.88 ** (epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# get the edge index of a trajectory considering the connectivity of whole set of trajectories
# the set of nodes in this input must be the mapped ids in the graph
def get_edge_index(nodes, adj_list):
    node_id_map = {}
    node_id = 0
    edge_index = [[],[]]
    for node in nodes:

        # If node ID not yet in map, add to map and increment node ID counter
        if node not in node_id_map.keys():
            node_id_map[node] = node_id
            node_id += 1

    for node in nodes:
        # Add edge from current node to previous node (if applicable) in current sentence
        adj = adj_list[node]
        for item in adj:
            if item not in node_id_map.keys():
                continue
            else:
                edge_index[0].append(node_id_map[node])
                edge_index[1].append(node_id_map[item])

    # Convert edge index to PyTorch tensor
    edge_index = torch.tensor(edge_index)

    return edge_index


def train():
    gc.collect()
    tempT, pointT, userT, seqlens, test_T, test_UserT, test_lens, User_List = read_train_data()  # 初始化参数 Python参数返回一一对应的，不能简单输出一个

    # # construct graph structure
    graph = GraphCenter(pointT, test_T, userT, test_UserT, spatio=500)
  
    graph.load_graphSet()

    # get poi one_hot encoding
    # poi_emb = poi_one_hot(graph.vocabs)
    n_poi = len(graph.vocabs)

    gnn = BiLSTM_GNN(n_input, n_hidden, n_classes, n_poi, gnn_tag=False, dropout=0.5)
    optimizer = optim.Adam(gnn.parameters(),
                       lr=learning_rate)

    for i in range(train_iters):
        step = 0
        acc = 0
        allAcc = 0
        print('#####################')
        gnn.train()
        while step < len(pointT):
            opt = optimizer
            opt.zero_grad()
            length = len(pointT[step])  #

            #xsx_step = [[get_pvector(x[0]) for x in [a for a in pointT[step]]]]  #
            xsx_step = [[graph.vocabs[x[0]] for x in [a for a in pointT[step]]]]  #

            xsy_step = [get_mask_index(userT[step], User_List)]

            xsx = torch.tensor(xsx_step).to(device)
            xsy = torch.tensor(xsy_step).to(device)

            xsx_map = [graph.vocabs[x[0]] for x in [a for a in pointT[step]]]

            edge_index = get_edge_index(xsx_map, graph.adj_lists)
            edge_index = torch.tensor(edge_index).to(device)

            out = gnn(xsx, edge_index)

            loss = F.cross_entropy(out, xsy)
            if step % dispaly_step == 0:
                print('step: ', step, 'loss: ', loss.item())
            loss.backward()
            opt.step()
            step += 1

        adjust_learning_rate(optimizer, i)

        test(gnn, pointT, userT, allAcc, User_List, graph, iter='train:' + str(i))
        print(loss.item())
        gnn.eval()
        test(gnn, test_T, test_UserT, allAcc, User_List, graph, iter='test:' + str(i))

    return


def test(gnn, test_T, test_U, allAcc, User_List, graph, iter, filename=result_out_path+result_output):
    ftestw = open(filename, 'a+')
    ftestw.write("iters:=" + str(iter) + " ")
    ftestw.write("allAcc:" + str(allAcc) + " ")
    step = 0
    acc = 0
    AccTop5 = 0
    AccTop10 = 0
    tempU = list(set(User_List))
    Dic = {}
    for i in range(len(tempU)):
        Dic[i] = [0, 0, 0, 0]  #
    while step < len(test_T):  #
        value = list()

        #xsx_step = [[get_pvector(x[0]) for x in [a for a in test_T[step]]]]  #
        xsx_step = [[graph.vocabs[x[0]] for x in [a for a in test_T[step]]]]
        xsx = torch.tensor(xsx_step).to(device)

        xsy_step = [get_mask_index(test_U[step], User_List)]  #
        xsy = torch.tensor(xsy_step).to(device)
        user_id = get_mask_index(test_U[step], User_List)

        xsx_map = [graph.vocabs[x[0]] for x in [a for a in test_T[step]]]

        edge_index = get_edge_index(xsx_map, graph.adj_lists)
        edge_index = torch.tensor(edge_index).to(device)

        Dic.get(user_id)[2] += 1  # a+c

        nowVec = gnn(xsx, edge_index)
        nowVec = nowVec[0]

        predictList = np.argpartition(a=-nowVec.detach().cpu().numpy(), kth=5)[:5]
        top10 = np.argpartition(a=-nowVec.detach().cpu().numpy(), kth=10)[:10]
        top1 = np.argpartition(a=-nowVec.detach().cpu().numpy(), kth=1)[:1]
        val = list(top1)
        print(val[0])
        Dic.get(val[0])[1] += 1  # a+b
        for i in range(len(top10)):
            value.append(get_true_index(top10[i], User_List))
        Test_10.append(value)

        for index in range(0, 5):
            if (predictList[index] == get_mask_index(test_U[step], User_List)):
                AccTop5 += 1
                break

        print(top1[0])
        if (top1[0] == xsy[0]):
            acc += 1
            Dic.get(user_id)[0] += 1  # a
        step += 1
    # Count Macro-F1
    macro = 0
    a = 0
    print(Dic)
    for i in list(Dic.keys()):
        if ((Dic.get(i)[1] + Dic.get(i)[2]) > 0):
            Dic.get(i)[3] = (2 * Dic.get(i)[0]) / (Dic.get(i)[1] + Dic.get(i)[2])
            macro += Dic.get(i)[3]
            a += Dic.get(i)[0]
    macro = macro / len(Dic)
    print('Dic length', len(Dic))
    try:
        ftestw.write("OUT CONSOLE: ")

        ftestw.write("step=" + str(step) + " ")
        ftestw.write(" length=" + str(len(test_T)) + " ")
        ftestw.write(" Accuracy1 numbers:" + str(acc) + " ")
        ftestw.write(" Accuracy5 numbers:" + str(AccTop5) + " ")
        ftestw.write(" Accuracy1:" + str(acc / step) + " ")
        ftestw.write(" Accuracy5:" + str(AccTop5 / step) + " ")
        ftestw.write(" Macro-F1:" + str(macro))
        ftestw.write('\n')
    except Exception:
        print('get error in count acc')
    ftestw.close()
    print("OUT_PUT accuraccy1=", acc)
    print("OUT_PUT accuraccy5=", AccTop5)
    print("Macro-F1", macro, 'A=', a)
    # Test_acc.append(acc / step)
    # Test_acc5.append(AccTop5 / step)
    # Test_macro.append(macro)
    # print Dic
    return 0


if __name__ == "__main__":
    #get_xs()
    t1=time.time()
    print("running data reader...")
    
    print("start training...")

    train()
    t2=time.time()

    print('total time cost:',(t2-t1)/3600)

