"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal / Max-Planck-Gesellschaft (MPG)
Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME
More information: http://flame.is.tue.mpg.de
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues, lbs, vertices2landmarks
from smplx.utils import Struct, rot_mat_to_euler, to_np, to_tensor

# chumpy objects in the FLAME pickle still import aliases removed by recent NumPy.
for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


class FLAME(nn.Module):
    """
    Differentiable FLAME decoder that outputs a mesh and 3D facial landmarks
    given shape, expression, and pose parameters.
    """

    def __init__(self, config):
        super().__init__()

        with open(config.flame_model_path, "rb") as f:
            flame_model = Struct(**pickle.load(f, encoding="latin1"))

        self.NECK_IDX = 1
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.use_face_contour = config.use_face_contour
        self.use_3D_translation = config.use_3D_translation

        self.register_buffer(
            "faces_tensor",
            to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )

        # Shape betas: first N are optimizable, the rest are fixed to zero
        default_shape = torch.zeros(
            [self.batch_size, 300 - config.shape_params],
            dtype=self.dtype, requires_grad=False,
        )
        self.register_parameter(
            "shape_betas", nn.Parameter(default_shape, requires_grad=False)
        )

        # Expression betas: same strategy
        default_exp = torch.zeros(
            [self.batch_size, 100 - config.expression_params],
            dtype=self.dtype, requires_grad=False,
        )
        self.register_parameter(
            "expression_betas", nn.Parameter(default_exp, requires_grad=False)
        )

        default_eye_pose = torch.zeros(
            [self.batch_size, 6], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eye_pose, requires_grad=False)
        )

        default_head_pose = torch.zeros(
            [self.batch_size, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "head_pose", nn.Parameter(default_head_pose, requires_grad=False)
        )

        default_transl = torch.zeros(
            [self.batch_size, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "transl", nn.Parameter(default_transl, requires_grad=False)
        )

        self.register_buffer(
            "v_template",
            to_tensor(to_np(flame_model.v_template), dtype=self.dtype),
        )

        shapedirs = flame_model.shapedirs
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))

        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Static landmark embeddings
        with open(config.static_landmark_embedding_path, "rb") as f:
            static_embeddings = Struct(**pickle.load(f, encoding="latin1"))

        lmk_faces_idx = static_embeddings.lmk_face_idx.astype(np.int64)
        self.register_buffer(
            "lmk_faces_idx", torch.tensor(lmk_faces_idx, dtype=torch.long)
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(static_embeddings.lmk_b_coords, dtype=self.dtype),
        )

        # Dynamic contour landmark embeddings (optional)
        if self.use_face_contour:
            contour_embeddings = np.load(
                config.dynamic_landmark_embedding_path,
                allow_pickle=True, encoding="latin1",
            )[()]
            dynamic_lmk_faces_idx = torch.tensor(
                np.array(contour_embeddings["lmk_face_idx"]).astype(np.int64),
                dtype=torch.long,
            )
            self.register_buffer("dynamic_lmk_faces_idx", dynamic_lmk_faces_idx)

            dynamic_lmk_bary_coords = torch.tensor(
                np.array(contour_embeddings["lmk_b_coords"]), dtype=self.dtype,
            )
            self.register_buffer("dynamic_lmk_bary_coords", dynamic_lmk_bary_coords)

            neck_kin_chain = []
            curr_idx = torch.tensor(self.NECK_IDX, dtype=torch.long)
            while curr_idx != -1:
                neck_kin_chain.append(curr_idx)
                curr_idx = self.parents[curr_idx]
            self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # pkl 数据已全部转为 buffer/parameter，释放原始 Struct 占用的 ~50MB 内存
        del flame_model

    def _find_dynamic_lmk_idx_and_bcoords(
        self, vertices, pose,
        dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
        neck_kin_chain, dtype=torch.float32,
    ):
        """
        Selects the face contour depending on the relative position of the head.
        Source: Modified for batches from https://github.com/vchoutas/smplx
        """
        batch_size = vertices.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3)).view(batch_size, -1, 3, 3)

        rel_rot_mat = (
            torch.eye(3, device=vertices.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)
        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)

        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def forward(self, shape_params=None, expression_params=None,
                pose_params=None, eye_pose=None, transl=None):
        """
        Input:
            shape_params:      (N, S) shape coefficients
            expression_params: (N, E) expression coefficients
            pose_params:       (N, P) pose coefficients
        Return:
            vertices:  (N, V, 3)
            landmarks: (N, K, 3)
        """
        betas = torch.cat(
            [shape_params, self.shape_betas, expression_params, self.expression_betas],
            dim=1,
        )
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        full_pose = torch.cat([self.head_pose, pose_params, eye_pose], dim=1)
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents, self.lbs_weights,
        )

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).repeat(self.batch_size, 1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            self.batch_size, 1, 1
        )

        if self.use_face_contour:
            dyn_lmk_faces_idx, dyn_lmk_bary_coords = (
                self._find_dynamic_lmk_idx_and_bcoords(
                    vertices, full_pose,
                    self.dynamic_lmk_faces_idx,
                    self.dynamic_lmk_bary_coords,
                    self.neck_kin_chain,
                    dtype=self.dtype,
                )
            )
            lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
            lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )

        if self.use_3D_translation:
            landmarks += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        return vertices, landmarks
