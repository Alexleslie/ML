import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', fc='0.8')   # 判断结点
leaf_node = dict(boxstyle='round4',fc='0.8')    # 叶子结点
arrow_args = dict(arrowstyle='<-')  # 箭头


def plot_node(node_txt, center_pt, parent_pt, node_type):  # 画图
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node ', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leaf(my_tree):  # 获得叶子结点数目
    num_leafs = 0  # 叶子结点数目
    first_str = list(my_tree.keys())[0]   # 获取第一个键值， 需要转列表
    second_dict = my_tree[first_str]  # 获取嵌套字典
    for key in second_dict.keys():  # 对于内嵌字典
        if type(second_dict[key]).__name__ == 'dict':  # 如果是字典而是而不是值（类别），则继续递归
            num_leafs += get_num_leaf(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):  # 获得树深度
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_mid_test(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leaf(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.total_w,
               plot_tree.yOff)
    plot_mid_test(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff),
                      cntr_pt, leaf_node)
            plot_mid_test((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff = 1.0/plot_tree.total_d


def create_plot_tree(in_tree):
    fig = plt.figure(1, facecolor='white',figsize=(8,8))
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leaf(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5/plot_tree.total_w
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()

