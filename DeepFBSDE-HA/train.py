import time
from solver import FeedForwardModel
import logging
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = False

'''全局绘图参数定义'''
linewidth = 0.5  # 绘图中曲线宽度
fontsize = 5  # 绘图字体大小
markersize = 2.5  # 标志中字体大小
legend_font_size = 5  # 图例中字体大小

def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Created successfully')
        return True
    else:
        print(path + ' Directory already exists')
        return False


def save_figure(dir, name):
    mkdir(dir)
    plt.savefig(dir + name, bbox_inches='tight')


def plot_result(alpha_list, delta_z_list, theta_list, q_list, theta_desire_list, figure_number=4):
    '''绘图参数定义'''
    label = ["Constrained_FC", "Unconstrained_FC", "Constrained_LSTM", "Unconstrained_LSTM"]
    color = ["r", "b", "g", "k"]
    line_style = ["-", "-", "-", "-"]

    '''绘制alpha曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Attack Angle $(Degree)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(alpha_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "alpha_Curve.png")
    plt.show()

    '''绘制delta_z曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Elevator $(Degree)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(delta_z_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "delta_z_Curve.png")
    plt.show()

    '''绘制theta曲线'''
    plt.figure(figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Pitch Angle $(Degree)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(theta_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.plot(theta_desire_list[0], label="$\\theta_{target}$", linestyle="--", linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "theta_Curve.png")
    plt.show()

    '''绘制q曲线'''
    plt.figure(num=None, figsize=(2.8, 1.7), dpi=300)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize)
    plt.xlabel("Time$(0.02s)$", fontproperties='Times New Roman', fontsize=fontsize)
    plt.ylabel("Pitch angular velocity $(Degree/s)$", fontproperties='Times New Roman', fontsize=fontsize)
    for i in range(figure_number):
        plt.plot(q_list[i], label=label[i], color=color[i], linestyle=line_style[i], linewidth=linewidth)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': legend_font_size})
    save_figure("./photo/exp1/", "q_Curve.png")
    plt.show()


def train(config, fbsde, save_best=True):
    logging.basicConfig(level=logging.INFO, format='%(levelname)-6s %(message)s')

    if fbsde.y_init:
        logging.info('Y0_true: %.4e' % fbsde.y_init)

    net = FeedForwardModel(config, fbsde)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr_value, weight_decay=config.weight_decay)
    start_time = time.time()
    training_history = []
    best_terminal_loss = float('+inf')

    dw_valid = fbsde.sample(config.valid_size)
    for step in range(config.num_iterations + 1):
        if step % config.logging_frequency == 0:
            net.train(False)
            loss, init, yr, ye, totalx, totalu = net(dw_valid)
            x_sample = totalx[-1]
            terminal_cost = (torch.mean(ye[:, 0])).detach().numpy()
            elapsed_time = time.time() - start_time
            training_history.append([step, loss, init.item(), elapsed_time])
            if config.verbose:
                logging.info(
                    "step: %5u, loss: %.4e, Y0: %.4e, terminal alpha: %.4e, terminal theta: %.4e,terminal q: %.4e, terminal cost: %.4e, elapsed time: %3u" % (
                        step, loss, init.item(), torch.mean(x_sample[:, 0, 0]), torch.mean(x_sample[:, 1, 0]),
                        torch.mean(x_sample[:, 2, 0]), terminal_cost, elapsed_time))
            # 根据终止状态代价最小，选择最好的模型
            if terminal_cost < best_terminal_loss:
                best_terminal_loss = terminal_cost
                if save_best:
                    print("model saved to", config.model_save_path)
                    torch.save(net, config.model_save_path)
        dw_train = fbsde.sample(config.batch_size)
        net.train(True)
        optimizer.zero_grad()
        loss, init, yr, ye, totalx, totalu = net(dw_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config.max_grad_norm)
        optimizer.step()
    training_history = np.array(training_history)

    if fbsde.y_init:
        logging.info('relative error of Y0: %s',
                     '{:.2%}'.format(abs(fbsde.y_init - training_history[-1, 2]) / fbsde.y_init))

    np.savetxt('{}_training_history.csv'.format(fbsde.__class__.__name__),
               training_history,
               fmt=['%d', '%.5e', '%.5e', '%d'],
               delimiter=",",
               header="step,loss_function,target_value,elapsed_time",
               comments='')


def valid(config, fbsde, path):
    net = torch.load(path)
    net.eval()
    # dw_valid = fbsde.sample(config.valid_size)
    dw_valid = torch.zeros([config.valid_size, config.dim, fbsde.num_time_interval])
    loss, init, yr, ye, totalx, totalu = net(dw_valid)
    return totalx, totalu


def plot(cfg, fbsde):
    path = ["Constrained_FC.pth", "Unconstrained_FC.pth", "cons_lstm.pth", "un_lstm.pth"]
    alpha = []
    theta = []
    q = []
    theta_desire = []
    delta_z = []
    for p in path:
        if p == "Constrained_FC.pth":
            cfg.constrained = True
            cfg.lstm = False
        if p == "Unconstrained_FC.pth":
            cfg.constrained = False
            cfg.lstm = False
        if p == "cons_lstm.pth":
            cfg.constrained = True
            cfg.lstm = True
        if p == "un_lstm.pth":
            cfg.constrained = False
            cfg.lstm = True

        x_sample, u = valid(cfg, fbsde, p)

        state = []
        for i in range(len(x_sample)):
            state.append(torch.mean(x_sample[i], dim=0))

        alpha_list = []
        for i in range(len(x_sample)):
            s = state[i][0].detach().numpy()
            alpha_list.append(state[i][0])
        alpha.append(alpha_list)

        theta_list = []
        for i in range(len(x_sample)):
            s = state[i][1].detach().numpy()
            theta_list.append(s)
        theta.append(theta_list)

        q_list = []
        for i in range(len(x_sample)):
            s = state[i][2].detach().numpy()
            q_list.append(s)
        q.append(q_list)

        theta_desire_list = []
        for i in range(len(x_sample)):
            theta_desire_list.append(10)
        theta_desire.append(theta_desire_list)

        delta_z_list = []
        for i in range(len(x_sample)):
            s = u[i][0][0].detach().numpy()
            delta_z_list.append(s)
        delta_z.append(delta_z_list)

    plot_result(alpha, delta_z, theta, q, theta_desire, figure_number=4)


if __name__ == '__main__':
    from config import get_config
    from equation import get_equation

    cfg = get_config('Aircraft')
    fbsde = get_equation('Aircraft', cfg.dim, cfg.total_time, cfg.delta_t)
    # train(cfg, fbsde)
    plot(cfg, fbsde)
