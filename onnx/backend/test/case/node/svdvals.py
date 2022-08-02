# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class SvdVals(Base):

    @staticmethod
    def export() -> None:
        dimensions = [
            ((3, 4), '2d'),
            ((2, 3, 4), '3d'),
            ((5, 4, 3, 2, 1, 3, 4), '7d')
        ]

        for (dims, dims_name) in dimensions:
            node = onnx.helper.make_node(
                'SVDVals',
                inputs=['A'],
                outputs=['S']
            )

            A = np.random.randn(*dims)

            S = np.linalg.svd(A, compute_uv=False)

            expect(node, inputs=[A], outputs=[S], name='test_svdvals_' + dims_name)
