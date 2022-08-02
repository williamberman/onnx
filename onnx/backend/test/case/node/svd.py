# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Svd(Base):

    @staticmethod
    def export() -> None:
        dimensions = [
            ((3, 4), '2d'),
            ((2, 3, 4), '3d'),
            ((5, 4, 3, 2, 1, 3, 4), '7d')
        ]

        for (dims, dims_name) in dimensions:
            full_node = onnx.helper.make_node(
                'SVD',
                inputs=['A'],
                outputs=['U', 'S', 'Vh']
            )
            full_node_explicit = onnx.helper.make_node(
                'SVD',
                inputs=['A'],
                outputs=['U', 'S', 'Vh'],
                full_matrices=1
            )
            partial_node = onnx.helper.make_node(
                'SVD',
                inputs=['A'],
                outputs=['U', 'S', 'Vh'],
                full_matrices=0
            )

            A = np.random.randn(*dims)

            full_results = np.linalg.svd(A)
            partial_results = np.linalg.svd(A, full_matrices=False)

            expect(full_node, inputs=[A], outputs=list(full_results), name='test_svd_' + dims_name)
            expect(full_node_explicit, inputs=[A], outputs=list(full_results), name='test_svd_' + dims_name + '_full_manually_set')
            expect(partial_node, inputs=[A], outputs=list(partial_results), name='test_svd_' + dims_name + '_partial')
