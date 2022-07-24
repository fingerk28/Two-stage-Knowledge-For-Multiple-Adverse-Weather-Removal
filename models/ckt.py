import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CKTModule(nn.Module):
    def __init__(self, channel_t, channel_s, channel_h, n_teachers):
        super().__init__()
        self.teacher_projectors = TeacherProjectors(channel_t, channel_h, n_teachers)
        self.student_projector = StudentProjector(channel_s, channel_h)
    
    def forward(self, teacher_features, student_feature):
        teacher_projected_feature, teacher_reconstructed_feature = self.teacher_projectors(teacher_features)
        student_projected_feature = self.student_projector(student_feature)

        return teacher_projected_feature, teacher_reconstructed_feature, student_projected_feature


class TeacherProjectors(nn.Module):
    """
    This module is used to capture the common features of multiple teachers.
    **Parameters:**
        - **channel_t** (int): channel of teacher features
        - **channel_h** (int): channel of hidden common features
    """
    def __init__(self, channel_t, channel_h, n_teachers):
        super().__init__()
        self.PFPs = nn.ModuleList()
        for _ in range(n_teachers):
            self.PFPs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channel_t, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=channel_h, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
                )
            )

        self.IPFPs = nn.ModuleList()
        for _ in range(n_teachers):
            self.IPFPs.append(
                nn.Sequential(
                    nn.Conv2d(channel_h, channel_t, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel_t, channel_t, kernel_size=3, stride=1, padding=1, bias=False)
                )
            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, features):
        assert len(features) == len(self.PFPs)

        projected_features = [self.PFPs[i](features[i]) for i in range(len(features))]
        reconstructed_features = [self.IPFPs[i](projected_features[i]) for i in range(len(projected_features))]

        return projected_features, reconstructed_features


class StudentProjector(nn.Module):
    """
    This module is used to project the student's features to common feature space.
    **Parameters:**
        - **channel_s** (int): channel of student features
        - **channel_h** (int): channel of hidden common features
    """
    def __init__(self, channel_s, channel_h):
        super().__init__()
        self.PFP = nn.Sequential(
            nn.Conv2d(in_channels=channel_s, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel_h, out_channels=channel_h, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, fs):
        projected_features = self.PFP(fs)

        return projected_features

