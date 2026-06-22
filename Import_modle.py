import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import time





def cosine_similarity(features, cluster_centers):
    # 计算余弦相似度
    # features: [N, D], centers: [K, D]
    features_norm = F.normalize(features, p=2, dim=1)
    centers_norm = F.normalize(cluster_centers, p=2, dim=1)
    similarity = torch.mm(features_norm, centers_norm.t())
    return similarity  # 返回 N x K 的相似度矩阵


class ContrastiveKMeansModel(nn.Module):
    def __init__(self, feature_dim, num_clusters, T=0.05):
        super(ContrastiveKMeansModel, self).__init__()
        self.T = T
        self.num_clusters = num_clusters
        # 增加 n_init=1 稍微加快速度，但在 forward 中跑 kmeans 依然是瓶颈
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=1)

    def forward(self, features):
        # 1. 使用 K-means 对特征进行聚类
        # 注意：detach() 意味着聚类过程不传导梯度，这是标准的 Deep Clustering 做法
        features_numpy = features.detach().cpu().numpy()
        start = time.time()
        # print(f"[ContrastiveKMeans] fitting KMeans: samples={features_numpy.shape[0]}, dim={features_numpy.shape[1]}, num_clusters={self.num_clusters}")
        self.kmeans.fit(features_numpy)
        # print(f"[ContrastiveKMeans] KMeans.fit done in {time.time()-start:.3f}s")

        # 2. 将聚类中心转回 Tensor
        cluster_centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float).to(features.device)

        # 3. 计算特征与所有聚类中心的余弦相似度 [N, K]
        similarity = cosine_similarity(features, cluster_centers)

        # 4. 获取伪标签 (Hard Assignment)
        # 虽然我们用硬标签做索引，但我们会用置信度来加权
        labels_assignment = torch.tensor(self.kmeans.labels_, device=features.device)

        # 5. 构建 Logits
        # 正样本对：样本 vs 分配到的中心
        l_pos = similarity[range(similarity.shape[0]), labels_assignment].unsqueeze(-1)

        # 负样本对：样本 vs 其他中心
        # 使用 mask 方式比循环拼接更高效且易读
        mask = torch.ones_like(similarity, dtype=torch.bool)
        mask[range(similarity.shape[0]), labels_assignment] = False
        l_neg = similarity[mask].view(similarity.shape[0], -1)

        # 拼接 logits [N, K] (第0列是正样本)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # 6. === 关键优化：计算样本置信度权重 ===
        # 逻辑：如果样本距离其中心很近，similarity 很大，权重接近 1
        # 如果样本在边界，similarity 较小，权重降低
        # 这里使用 Softmax 的最大概率作为置信度指标
        probs = F.softmax(similarity / self.T, dim=1)
        confidence, _ = probs.max(dim=1)  # [N]

        # 进一步过滤：对于置信度极低的样本（噪声），可以将其权重设为极小
        # 例如：weight = confidence^2 或者是简单的线性 confidence
        sample_weights = confidence.detach()

        # 标签：正样本永远在 logits 的第 0 列
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)

        return logits, labels, sample_weights
