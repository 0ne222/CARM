import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.optim as optim
import numpy as np
import time
import scipy.io as sio
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from post_clustering import spectral_clustering, acc, nmi, f1_score, rand_index_score, adjusted_rand_score
from noise import add_noise_to_multiview_data
# 添加缺失的导入
from S1 import MVDSC
from Import_modle import ContrastiveKMeansModel
from Visualisation import (
    save_original_data_visualization,
    save_tsne_visualization,
    save_c_matrix_visualization,
    save_c_matrix_visualization_sorted,
    save_cluster_vs_true_sankey,
    save_convergence_visualization,
)
def _feature_to_similarity(features, eps=1e-8):
        """Convert feature matrix to a symmetric similarity matrix for visualization."""
        features = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + eps)
        sim = np.matmul(features, features.T)
        sim = 0.5 * (sim + sim.T)
        return sim


def save_reliability_histogram(reliability, save_dir, db_name):
    """Save reliability-score density curve and return file path."""
    rel_np = reliability.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{db_name}_reliability_curve.png")

    x_min = float(rel_np.min())
    x_max = float(rel_np.max())
    if x_min == x_max:
        x_min -= 1e-3
        x_max += 1e-3
    x_grid = np.linspace(x_min, x_max, 512)

    if rel_np.size > 1 and np.std(rel_np) > 1e-12:
        density = gaussian_kde(rel_np)
        y_grid = density(x_grid)
    else:
        y_grid = np.zeros_like(x_grid)
        y_grid[len(y_grid) // 2] = 1.0

    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, y_grid, color='skyblue', linewidth=2.5)
    plt.fill_between(x_grid, y_grid, color='skyblue', alpha=0.25)
    plt.xlabel('Reliability Score')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path, rel_np


def save_P_and_reliability_visualization(P, reliability, save_dir, db_name):
    """Save a joint visualization of P and reliability."""
    P_np = P.detach().cpu().numpy()
    rel_np = reliability.detach().cpu().numpy()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{db_name}_P_and_reliability.png")

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(P_np, cmap='magma', aspect='auto')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Samples')
    plt.colorbar(im, ax=ax1, label='Strength')

    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(range(len(rel_np)), rel_np, alpha=0.6, color='blue', s=10)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Reliability')
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path, P_np, rel_np


def train(model,  
          x1, x2, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, noise_ratio=0, show=10,
          vis_dir='results/tsne', matrix_vis_dir='results/c_matrix',
          sorted_matrix_vis_dir='results/c_matrix_sorted', db_name='dataset',
          base_save_dir='results', legend_loc='lower right', legend_anchor_x=None, legend_anchor_y=None,
          show_legend=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x1, torch.Tensor):
        x1 = torch.tensor(x1, dtype=torch.float32, device=device)
    x1 = x1.to(device)
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, dtype=torch.float32, device=device)
    x2 = x2.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))

    # 确保 y 是一维数组
    y = np.squeeze(y)

    # 使用新的多视图噪声添加函数（噪声比例由调用方传入）
    views = [x1, x2]
    noisy_views = add_noise_to_multiview_data(views, noise_ratio)
    x1, x2 = noisy_views[0], noisy_views[1]
    print(f"Added noise with ratio: {noise_ratio}")

    epoch_history = []
    loss_history = []
    acc_history = []
    nmi_history = []
    ari_history = []
    # 初始化 contrastive 模型，使其使用训练数据的真实聚类数 K
    try:
        feature_dim = model.ae.enlayer1_2[0].out_features
    except Exception:
        # 备用方案：取第一个视图编码后的维度
        feature_dim = model.ae.enlayer1_2[0].out_features if hasattr(model.ae.enlayer1_2[0], 'out_features') else None
    if feature_dim is None:
        # 如果仍然无法获取，在第一次前向时 MVDSC 会创建一个默认的 contrastive 模型
        pass
    else:
        model.contrastive = ContrastiveKMeansModel(feature_dim=feature_dim, num_clusters=K)
        model.contrastive.to(device)
    # 总计时开始
    total_start_time = time.time()
    # 记录上次打印的起始时间（第一次计时从训练开始）
    cycle_start_time = time.time()
    for epoch in range(epochs):
 
        # 解包模型输出（共 16 项，含 concat contrastive 输出；后 3 项 loss_fn 不使用）
        x1_recon, z1, z1_recon, \
        x2_recon, z2, z2_recon, z_concatenated, \
        logits1, labels1, logits2, labels2, \
        weights1, weights2, r_cross = model(x1, x2)
        
        # 调用 loss_fn（当前版本只使用两视图的 contrastive 输出）
        loss = model.loss_fn(x1, x2, 
                 x1_recon, z1, z1_recon,
                 x2_recon, z2, z2_recon, z_concatenated,
                 logits1=logits1, labels1=labels1, logits2=logits2, labels2=labels2,
                 weights1=weights1, weights2=weights2, r_cross=r_cross,
                 weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print at epoch 0, show, 2*show, ... and also at final epoch
        if epoch % show == 0 or epoch == epochs - 1:
            cycle_time = time.time() - cycle_start_time  # 计算自上次打印以来的耗时
            cycle_start_time = time.time()  # 重置起始时间，供下一次打印使用
 
            C = model.get_C()
            
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            # 确保 y_pred 是一维数组
            y_pred = np.squeeze(y_pred)
 
            f_measure = f1_score(y, y_pred)
            ri = rand_index_score(y, y_pred)
            ar = adjusted_rand_score(y, y_pred)
            Acc = acc(y, y_pred)
            Nmi = nmi(y, y_pred)
            loss = loss.item() / y_pred.shape[0]
            print("epoch: %d" % epoch, \
                  "loss:%.4f" % loss, \
                  "ACC: %.4f" % Acc, \
                  "NMI: %.4f" % Nmi, \
                  "ARI: %.4f" % ar, \
                  "F-measure: %.4f" % f_measure, \
                  "time: %.4fs" % cycle_time)  # 简化时间显示信息
            # epoch_history.append(epoch)
            # loss_history.append(loss)
            # acc_history.append(Acc)
            # nmi_history.append(Nmi)
            # ari_history.append(ar)
            # vis_path = save_tsne_visualization(C, y_pred, epoch, vis_dir, db_name=db_name)
            # cmat_path = save_c_matrix_visualization(C, epoch, matrix_vis_dir, db_name=db_name)
            # cmat_sorted_path = save_c_matrix_visualization_sorted(C, y_pred, epoch, sorted_matrix_vis_dir, db_name=db_name)
            # save_cluster_vs_true_sankey(y_pred,y,epoch,save_dir=os.path.join(base_save_dir, 'sankey', db_name),)
            # print(f"t-SNE saved: {vis_path}")
            # print(f"C heatmap saved: {cmat_path}")
            # print(f"C sorted heatmap saved: {cmat_sorted_path}")
            # 仅记录收敛历史，避免在训练循环中频繁绘图导致速度下降
    # 训练全部完成，打印总耗时
    total_time = time.time() - total_start_time
    print("总训练时间: %.4fs" % total_time)

    # 训练结束后导出 soft_P_from_C 的 reliability 并绘制分布直方图
    with torch.no_grad():
        P, reliability = model.soft_P_from_C()
    reliability_dir = os.path.join(base_save_dir, 'reliability', db_name)
    reliability_path, rel_np = save_reliability_histogram(reliability, reliability_dir, db_name)
    reliability_npy_path = os.path.join(reliability_dir, f"{db_name}_reliability.npy")
    np.save(reliability_npy_path, rel_np)
    joint_vis_path, P_np, rel_np = save_P_and_reliability_visualization(P, reliability, reliability_dir, db_name)
    P_npy_path = os.path.join(reliability_dir, f"{db_name}_P.npy")
    np.save(P_npy_path, P_np)
    print(f"Reliability histogram saved: {reliability_path}")
    print(f"P and reliability visualization saved: {joint_vis_path}")
    print(f"P numpy saved: {P_npy_path}")
   

    # # 训练结束后一次性绘制收敛曲线，保证结果可复现且不阻塞训练
    # if len(epoch_history) > 0:
    #     convergence_dir = os.path.join(base_save_dir, 'convergence', db_name+ "-1")
    #     convergence_path = save_convergence_visualization(
    #         epoch_history=epoch_history,
    #         loss_history=loss_history,
    #         acc_history=acc_history,
    #         nmi_history=nmi_history,
    #         ari_history=ari_history,
    #         save_dir=convergence_dir,
    #         db_name=db_name,
    #         show_legend=show_legend,
    #         legend_loc=legend_loc,
    #         legend_anchor_x=legend_anchor_x,
    #         legend_anchor_y=legend_anchor_y,
    #         grid_alpha=0.8,
    #     )
    #     print(f"Convergence curve saved: {convergence_path}")


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='MVDSC - 2 Views')
    parser.add_argument('--db', default='Caltech101-20',
                        choices=['HandWritten','WebKB', 'Caltech101', 'Caltech101-7', 'Scene-15', 'Caltech101-20'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--legend-loc', default='lower right', type=str)
    parser.add_argument('--legend-anchor-x', default=None, type=float)
    parser.add_argument('--legend-anchor-y', default=None, type=float)
    parser.add_argument('--show-legend', action='store_true')
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    db = args.db

    if db == 'Scene-15':
        data = sio.loadmat('dataset/Scene-15-0.5.mat')
        print(data.keys())
        features = data['X']
        y = data['Y']
        y = np.squeeze(y - 1)
        views = []        
        K = len(np.unique(y))
        print('K =', K)
        for v in features[0]:
            views.append(v)
        print(f"views: {len(views)}")
    
        x1 = views[0]
        x2 = views[1]
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")

        dim1 = [x1.shape[1], 2000, 500]  #C1-0.5[x1.shape[1], 2000, 500] 2 0.8 0.8 15 4 acc：0.6
        print(f"dim1[0]: {x1.shape[1]}") #C1-0.2[x1.shape[1], 2000, 500] 2 0.8 0.8 15 4 acc：0.6
        dim2 = [x2.shape[1], 2000, 500]  #C1-0.4[x1.shape[1], 2000, 500] 2 0.8 0.2 15 4 acc：0.6
        print(f"dim2[0]: {x2.shape[1]}") #C1-0.6[x1.shape[1], 2000, 500] 2 0.8 0.4 15 4 acc：0.6
        epochs = 50                     #C1-0.8[x1.shape[1], 2000, 500] 2 0.8 0.2 15 4 acc：0.6
        weight_coef = 2                  #C1[x1.shape[1], 2000, 500] 2 0.8 0.8 15 4 acc：0.6
        weight_selfExp = 0.8
        alpha = 0.8  # threshold of C
        dim_subspace = 15 # dimension of each subspace
        ro = 4 #
        noise_ratio = 0
    
    if db == 'HandWritten':
        data = sio.loadmat('dataset/Handwritten_numerals-0.5.mat')
        print(data.keys())
        
        features = data['X']
        print(f"features.shape: {features.shape}")  # (6, 1)
        
        y = data['Y']
        y = np.squeeze(y - 1)      
        K = len(np.unique(y))
        print('K =', K)

        views = []
        for v in features:  
            views.append(v[0])  # ← 加这一层 [0]
        print(f"views: {len(views)}")  # 6 ✓
        for i, v in enumerate(views):
            print(f"  view[{i}] shape: {v.shape}")
        x1 = views[0]   # (2000, 216) ✓
        x2 = views[4]   # (2000, 240) ✓
        
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")  # 2000 ✓

        dim1 = [x1.shape[1], 2000, 500]   #C1[x1.shape[1], 2000, 500] 0.2 0.8 0.2 10 4 acc0.83
        print(f"dim1[0]: {x1.shape[1]}")  #C1-0.5[x1.shape[1], 2000, 500] 0.2 0.8 0.2 10 4 acc0.975
        dim2 = [x2.shape[1], 2000, 500]   #C1-0.2[x1.shape[1], 2000, 500] 0.2 0.8 0.2 10 4 acc：0.6
        print(f"dim2[0]: {x2.shape[1]}")  #C1-0.4[x1.shape[1], 2000, 500] 2 0.8 0.2 10 4 acc：0.6
        epochs = 40                      #C1-0.6[x1.shape[1], 2000, 500] 2 0.8 0.2 10 4 acc：0.6
        weight_coef = 0.2
        weight_selfExp = 0.8

        noise_ratio = 0
        alpha = 0.2  # threshold of C
        dim_subspace = 10 # dimension of each subspace
        ro = 4 #


    if db == 'Caltech101-20':

        data = sio.loadmat('dataset/Caltech101-20-0.5.mat')
        print(data.keys())
        # pdb.set_trace()
        features = data['X']
        y = data['Y']
        y = np.squeeze(y - 1)
        K = len(np.unique(y))
        print('K =', K)
        views = []
        view_shape = []
        for v in features[0]:
            view_shape.append(v.shape[1])
            views.append(v)
        x1 = views[3]       # (2386, 1984)
        # x1 = np.transpose(x1, (1, 0))
        x2 = views[4]       # (2386, 512)
        # x2 = np.transpose(x2, (1, 0))
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")
        dim1 = [x1.shape[1], 1500, 800]      #C1 dim1 = [x1.shape[1], 1500, 800] 2 0.1 0.02 12/15 6 acc0.75
        dim2 = [x2.shape[1], 1500, 800]      #C1-0.5 dim1 = [x1.shape[1], 1500, 800/1500, 800] 1 0.1 0.15 15 6 acc0.75
        print(f"dim1: {dim1}, dim2: {dim2}") #C1-0.2 dim1 = [x1.shape[1], 1500, 500/1500, 500] 2 0.1 0.02 12 6 acc0.75
        epochs = 30                         #C1-0.4 dim1 = [x1.shape[1], 1200, 400/500, 400] 1 0.1 0.02 12 6 acc0.75
        weight_coef = 1                      #C1-0.6 dim1 = [x1.shape[1], 1500, 500/1500, 500] 1 0.1 0.2 15 6 acc0.75
        weight_selfExp = 0.1                #C1-0.8 dim1 = [x1.shape[1], 1200, 400/500, 400] 2 0.1 0.02 15 6 acc0.75
        noise_ratio = 0

        alpha = 0.2  # threshold of C
        dim_subspace = 15 # dimension of each subspace
        ro = 6 #

    if db == 'Caltech101-7':

        data = sio.loadmat('dataset/Caltech101-7.mat')
        print(data.keys())
        # pdb.set_trace()
        features = data['X']
        y = data['Y']
        y = np.squeeze(y - 1)
        K = len(np.unique(y))
        print('K =', K)
        views = []
        view_shape = []
        for v in features[0]:
            view_shape.append(v.shape[1])
            views.append(v)
        x1 = views[3] 
        x2 = views[4]     
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")
        dim1 = [x1.shape[1],1000, 450] #C1 dim1 = [x1.shape[1], 1000/500, 450] 2 0.1 0.02 12 5 acc0.94
        dim2 = [x2.shape[1],500, 450]
        print(f"dim1: {dim1}, dim2: {dim2}")
        epochs = 600
        weight_coef = 1
        weight_selfExp = 0.1
        noise_ratio = 0

        alpha = 0.02 # threshold of C
        dim_subspace = 12 # dimension of each subspace
        ro = 5 #


    if db == 'Caltech101':
        data = sio.loadmat('dataset/Caltech101-all-0.5.mat')
        print(data.keys())
        # pdb.set_trace()
        features = data['X']
        y = data['Y']
        y = np.squeeze(y - 1)
        K = len(np.unique(y))
        print('K =', K)
        views = []
        view_shape = []
        for v in features[0]:
            view_shape.append(v.shape[1])
            views.append(v)
        x1 = views[3] 
        x2 = views[4]     
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")
        dim1 = [x1.shape[1],1408, 64]       #C1     dim1 = [x1.shape[1], 1500, 500/500, 500] 2 0.2 0.02 40 5 acc0.75
        dim2 = [x2.shape[1], 512, 64]       #C1-0.4 dim1 = [x1.shape[1], 1200, 40/512, 40] 2 0.2 0.1 20 6 acc0.75
        print(f"dim1: {dim1}, dim2: {dim2}")#C1-0.5 dim1 = [x1.shape[1], 1408, 64/512, 64] 2 0.2 0.1 20 6 acc0.75
        epochs = 600
        weight_coef = 2
        weight_selfExp = 0.2
        noise_ratio = 0

        alpha = 0.1  # threshold of C
        dim_subspace = 20 # dimension of each subspace
        ro = 6 #

    if db == 'WebKB':
        # 加载mat文件
        data = sio.loadmat('dataset/WebKB-0.5.mat')  # 请确认实际文件路径
        print(data.keys())
        # 获取特征和标签
        features = data['X']
        y = data['gnd']
        # 处理标签（如果是1-based索引，转换为0-based）
        y = np.squeeze(y - 1) if y.min() == 1 else np.squeeze(y)
        # 获取视图数量
        views = []
        K = len(np.unique(y))
        print('K =', K)
        for v in features[0]:  # 根据Scene数据集的加载方式
            views.append(v)
        print(f"views: {len(views)}")
        x1 = views[0]
        x2 = views[1]
        
        num_sample = x1.shape[0]
        print(f"num_sample: {num_sample}")

        dim1 = [x1.shape[1], 2000, 500]  # C1 dim1 = [x1.shape[1], 2000, 500] 2 0.8 0.8 12 4 0.2 acc0.75
        print(f"dim1: {dim1}")           #C1-0.5 dim1 = [x1.shape[1], 2000, 500] 2 0.1 0.2 25 6 acc0.97
        dim2 = [x2.shape[1], 2000, 500]  #C1-0.2 dim1 = [x1.shape[1], 2000, 500] 2   1 0.8  2 6 acc0.97
        print(f"dim2: {dim2}")           #C1-0.4 dim1 = [x1.shape[1], 2000, 500] 2 0.1 0.8 25 6 acc0.97
                                         #C1-0.6 dim1 = [x1.shape[1], 1800, 400] 2 0.1 0.2  6 6 acc0.97
        # 设置超参数
        epochs = 800
        weight_coef = 2
        weight_selfExp = 0.1
        
        noise_ratio = 0
        alpha = 0.2  # threshold of C
        dim_subspace = 25  # dimension of each subspace
        ro = 6

    # raw_vis_dir = os.path.join(args.save_dir, 'original', db)
    # raw_vis_path_1 = save_original_data_visualization(x1, y, raw_vis_dir, db_name=f'{db}_view1')
    # raw_vis_path_2 = save_original_data_visualization(x2, y, raw_vis_dir, db_name=f'{db}_view2')
    # raw_vis_path_12 = save_original_data_visualization([x1, x2], y, raw_vis_dir, db_name=f'{db}_views12')
    # print(f"Original data scatter saved: {raw_vis_path_1}")
    # print(f"Original data scatter saved: {raw_vis_path_2}")
    # print(f"Original data scatter saved: {raw_vis_path_12}")

    MVDSC = MVDSC(num_sample = num_sample, dim1 = dim1, dim2 = dim2)
    MVDSC.to(device)
       
    ae_state_dict = torch.load('D:/AAA/project/WWW/pre_weight/%s-0.5-C1.pkl' % db, weights_only=True)
    MVDSC.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")

    train(MVDSC, x1, x2, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
            alpha=alpha, dim_subspace=dim_subspace, ro=ro, noise_ratio=noise_ratio,
            show=args.show_freq, device=device,
            vis_dir=os.path.join(args.save_dir, 'tsne', db),
            matrix_vis_dir=os.path.join(args.save_dir, 'c_matrix', db),
            sorted_matrix_vis_dir=os.path.join(args.save_dir, 'c_matrix_sorted', db),
            db_name=db,
            base_save_dir=args.save_dir,
            legend_loc=args.legend_loc,
            legend_anchor_x=args.legend_anchor_x,
            legend_anchor_y=args.legend_anchor_y,
            show_legend=args.show_legend)
    # torch.save(MVDSC.state_dict(), 'D:/AAA/project/WWW/pre_weight/%s-test1.pkl' % db)

#2000, 100 0.0 2/0.8/0.8/12/4 acc0.56
#2000, 100 0.2 2/0.8/0.8/12/4 acc0.55
#2000, 100 0.5 2/0.8/0.8/12/4 acc0.52
#2000, 500 0.8 2/0.8/0.8/12/4 acc0.54

 




 #现在介绍我的训练策略，第一次训练启用代码torch.save(MVDSC.state_dict(), 'D:/AAA/project/WWW/pre_weight/%s-test1.pkl' % db)；注释掉ae_state_dict = torch.load('D:/AAA/project/WWW/pre_weight/%s-0.5-C1.pkl' % db, weights_only=True)
    # MVDSC.load_state_dict(ae_state_dict)
    # print("Pretrained ae weights are loaded successfully.")，就能开始第一次训练。得到满意结果，就注释torch.save(MVDSC.state_dict(), 'D:/AAA/project/WWW/pre_weight/%s-test1.pkl' % db)，启用ae_state_dict = torch.load('D:/AAA/project/WWW/pre_weight/%s-0.5-C1.pkl' % db, weights_only=True)
    # MVDSC.load_state_dict(ae_state_dict)
    # print("Pretrained ae weights are loaded successfully.")，就能开始加载权重训练。如果第一次训练结果不好就再一次训练，直到有较好结果猜加载权重训练。这样的做法局限性在哪里