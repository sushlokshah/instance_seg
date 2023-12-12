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
        :param ground_truth: (N, 1, H, W)
        :return: (N, C, K)
        """
        N, C, H, W = features.size()
        feature = features.view(N, C, H * W)
        ground_truth = ground_truth.view(N, H * W)
        # make gt scaler
        ground_truth = ground_truth.to(torch.long)
        instances_batch = []
        cluster_mean_batch = []
        for n in range(N):
            instances = ground_truth[n].unique(sorted=True)
            # print(instances)
            instances_batch.append(instances)
            cluster_mean = []
            for i in range(len(instances)):
                cluster_mean.append(
                    torch.mean(feature[n] * (ground_truth[n] == instances[i]), dim=1)
                )
            cluster_mean = torch.stack(cluster_mean, dim=1)
            cluster_mean_batch.append(cluster_mean)
            # print(cluster_mean[-1].shape)
        return cluster_mean_batch, instances_batch

    def variance_loss(
        self, features, cluster_mean_batch, instances_batch, ground_truth
    ):
        """
        :param features: (N, C, H, W)
        :param cluster_mean: (N, C, K)
        :param ground_truth: (N, H, W)
        :return: variance loss
        """
        N, C, H, W = features.size()
        features = features.view(N, C, H * W)
        ground_truth = ground_truth.view(N, 1, H * W)
        variance_loss = 0
        for n in range(N):
            for k in range(len(instances_batch[n])):
                variance_loss += torch.max(
                    torch.mean(
                        torch.norm(
                            features[n] * (ground_truth[n] == instances_batch[n][k])
                            - cluster_mean_batch[n][:, k].unsqueeze(1),
                            dim=0,
                        ),
                        dim=0,
                    )
                    - self.delta_variance_loss,
                    torch.tensor(0).cuda(),
                ) / len(instances_batch[n])

        return variance_loss / N

    def distance_loss(self, cluster_mean):
        """
        :param cluster_mean: (N, C, K)
        :param ground_truth: (N, H, W)
        :return: distance loss
        """
        distance_loss = 0
        N = len(cluster_mean)

        for n in range(N):
            K = len(cluster_mean[n][0])
            # print(K)
            for k in range(len(cluster_mean[n][0])):
                # print(k)
                # print(cluster_mean[n][:, k].unsqueeze(1).repeat(1, K - 1).shape)
                # print(cluster_mean[n][:, :k].shape)
                # print(cluster_mean[n][:, k + 1 :].shape)
                # print(
                #     torch.cat(
                #         [cluster_mean[n][:, :k], cluster_mean[n][:, k + 1 :]], dim=1
                #     ).shape
                # )
                distance_loss += torch.max(
                    2 * self.delta_cluster_distance
                    - torch.mean(
                        torch.norm(
                            cluster_mean[n][:, k].unsqueeze(1).repeat(1, K - 1)
                            - torch.cat(
                                [cluster_mean[n][:, :k], cluster_mean[n][:, k + 1 :]],
                                dim=1,
                            ),
                            dim=0,
                        ),
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
        normalization_loss = 0
        N = len(cluster_mean)
        for n in range(len(cluster_mean)):
            for k in range(len(cluster_mean[n][0])):
                normalization_loss += torch.norm(cluster_mean[n][:, k], dim=0) / len(
                    cluster_mean[n][0]
                )
        return normalization_loss / (N)

    def forward(self, features, ground_truth):
        """
        :param features: (N, C, H, W)
        :param ground_truth: (N, H, W)
        :return: loss
        """
        N, C, H, W = features.size()
        cluster_mean, instances = self.calculate_cluster_mean(features, ground_truth)
        variance_loss = self.variance_loss(
            features, cluster_mean, instances, ground_truth
        )
        distance_loss = self.distance_loss(cluster_mean)
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


if __name__ == "__main__":
    loss = Cluster_loss()
    features = torch.randn(4, 128, 64, 64)
    ground_truth = torch.randint(0, 10, (4, 64, 64))
    clusters, instances = loss(features, ground_truth)
    for c, i in zip(clusters, instances):
        print(c.shape)
        print(i.shape)
