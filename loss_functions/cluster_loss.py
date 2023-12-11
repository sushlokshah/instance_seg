import torch
import torch.nn as nn
import torch.nn.functional as F


class Cluster_loss(nn.Module):
    def __init__(
        self,
        delta_variance_loss=0.2,
        delta_cluster_distance=0.2,
        alpha=1,
        beta=1,
        gamma=0.001,
    ):
        super(Cluster_loss, self).__init__()
        self.delta_variance_loss = delta_variance_loss
        self.delta_cluster_distance = delta_cluster_distance
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def calculate_cluster_mean(self, features, ground_truth):
        """
        :param features: (N, C, H, W)
        :param ground_truth: (N, H, W)
        :return: (N, C, K)
        """
        N, C, H, W = features.size()
        K = ground_truth.max() + 1
        cluster_mean = torch.zeros((N, C, K)).cuda()
        for n in range(N):
            for k in range(K):
                cluster_mean[n, :, k] = features[n, :, ground_truth[n] == k].mean(dim=1)

        return cluster_mean

    def variance_loss(self, features, cluster_mean, ground_truth):
        """
        :param features: (N, C, H, W)
        :param cluster_mean: (N, C, K)
        :param ground_truth: (N, H, W)
        :return: variance loss
        """
        N, C, H, W = features.size()
        K = ground_truth.max() + 1
        variance_loss = 0
        for n in range(N):
            for k in range(K):
                variance_loss += torch.max(
                    F.mse_loss(
                        features[n, :, ground_truth[n] == k],
                        cluster_mean[n, :, k].view(C, 1).repeat(1, H * W),
                    )
                    - self.delta_variance_loss,
                    torch.tensor(0).cuda(),
                )
        return variance_loss / (N * K)

    def distance_loss(self, cluster_mean, ground_truth):
        """
        :param cluster_mean: (N, C, K)
        :param ground_truth: (N, H, W)
        :return: distance loss
        """
        N, C, K = cluster_mean.size()
        distance_loss = 0
        for n in range(N):
            for k in range(K):
                distance_loss += torch.max(
                    2 * self.delta_cluster_distance
                    - torch.norm(
                        cluster_mean[n, :, k]
                        - cluster_mean[n, :, torch.arange(K) != k],
                        dim=0,
                    ),
                    torch.tensor(0).cuda(),
                )
        return distance_loss / (N * K)

    def normalization_loss(self, cluster_mean):
        """
        :param cluster_mean: (N, C, K)
        :return: normalization loss
        """
        N, C, K = cluster_mean.size()
        normalization_loss = 0
        for n in range(N):
            for k in range(K):
                normalization_loss += torch.norm(cluster_mean[n, :, k], dim=0)
        return normalization_loss / (N * K)

    def forward(self, features, ground_truth):
        """
        :param features: (N, C, H, W)
        :param ground_truth: (N, H, W)
        :return: loss
        """
        N, C, H, W = features.size()
        K = ground_truth.max() + 1
        cluster_mean = self.calculate_cluster_mean(features, ground_truth)
        variance_loss = self.variance_loss(features, cluster_mean, ground_truth)
        distance_loss = self.distance_loss(cluster_mean, ground_truth)
        normalization_loss = self.normalization_loss(cluster_mean)
        total_loss = (
            self.alpha * variance_loss
            + self.beta * distance_loss
            + self.gamma * normalization_loss
        )
        return total_loss, (
            variance_loss,
            distance_loss,
            normalization_loss,
            cluster_mean,
        )


# def Classification_loss(features, ground_truth):
#


class Classification_loss(nn.Module):
    def __init__(self):
        super(Classification_loss, self).__init__()

    def forward(self, cluster_head, ground_truth):
        """
        :param features: (N, C, H, W)
        :param ground_truth: (N, H, W)
        :return: classification loss
        """
        N, C, K = cluster_head.size()
        # K = ground_truth.max() + 1
        classification_loss = 0
        for n in range(N):
            for k in range(K):
                classification_loss += F.cross_entropy(
                    cluster_head[n, :, k], ground_truth[n, k]
                )
        return classification_loss / (N * K)
