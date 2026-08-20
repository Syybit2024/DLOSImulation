import unittest

import torch

from models.point_ssm.backbone import SerializedConv1d
from models.utils.structure import Point


def make_point(feat):
    point_count = feat.shape[0]
    order = torch.arange(point_count).unsqueeze(0)
    return Point(
        feat=feat,
        batch=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
        serialized_order=order,
        serialized_inverse=order.clone(),
    )


class SerializedConv1dTest(unittest.TestCase):
    def test_shape_gradient_and_batch_isolation(self):
        torch.manual_seed(0)
        layer = SerializedConv1d(2, 4, kernel_size=3)
        feat = torch.randn(8, 2, requires_grad=True)

        output = layer(make_point(feat.clone())).feat
        self.assertEqual(output.shape, (8, 4))
        output.sum().backward()
        self.assertIsNotNone(feat.grad)

        changed_feat = feat.detach().clone()
        changed_feat[5:] += 1000
        changed_output = layer(make_point(changed_feat)).feat
        torch.testing.assert_close(output[:5].detach(), changed_output[:5])


if __name__ == "__main__":
    unittest.main()
