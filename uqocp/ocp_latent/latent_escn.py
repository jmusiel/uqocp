
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.escn.so3 import (
    CoefficientMapping,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
)
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)
from ocpmodels.models.escn.escn import eSCN

try:
    from e3nn import o3
except ImportError:
    pass


@registry.register_model("new_latent_escn")
class latent_eSCN(eSCN):
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        device = data.pos.device
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype

        start_time = time.time()
        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        self.SO3_edge_rot = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_edge_rot.append(
                SO3_Rotation(edge_rot_mat, self.lmax_list[i])
            )

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l=0,m=0 coefficients for each resolution
        for i in range(self.num_resolutions):
            x.embedding[:, offset_res, :] = self.sphere_embedding(
                atomic_numbers
            )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # This can be expensive to compute (not implemented efficiently), so only do it once and pass it along to each layer
        mappingReduced = CoefficientMapping(
            self.lmax_list, self.mmax_list, device
        )

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            if i > 0:
                x_message = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

                # Residual layer for all layers past the first
                x.embedding = x.embedding + x_message.embedding

            else:
                # No residual for the first layer
                x = self.layer_blocks[i](
                    x,
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    self.SO3_edge_rot,
                    mappingReduced,
                )

        # Sample the spherical channels (node embeddings) at evenly distributed points on the sphere.
        # These values are fed into the output blocks.
        x_pt = torch.tensor([], device=device)
        offset = 0
        # Compute the embedding values at every sampled point on the sphere
        for i in range(self.num_resolutions):
            num_coefficients = int((x.lmax_list[i] + 1) ** 2)
            x_pt = torch.cat(
                [
                    x_pt,
                    torch.einsum(
                        "abc, pb->apc",
                        x.embedding[:, offset : offset + num_coefficients],
                        self.sphharm_weights[i],
                    ).contiguous(),
                ],
                dim=2,
            )
            offset = offset + num_coefficients

        x_pt = x_pt.view(-1, self.sphere_channels_all)

        latent_rep = x.embedding.mean(dim=-1)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x_pt)
        energy = torch.zeros(len(data.natoms), device=device)
        energy.index_add_(0, data.batch, node_energy.view(-1))
        # Scale energy to help balance numerical precision w.r.t. forces
        energy = energy * 0.001

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x_pt, self.sphere_points)

        if self.show_timing_info is True:
            torch.cuda.synchronize()
            logging.info(
                "{} Time: {}\tMemory: {}\t{}".format(
                    self.counter,
                    time.time() - start_time,
                    len(data.pos),
                    torch.cuda.max_memory_allocated() / 1000000,
                )
            )

        self.counter = self.counter + 1

        if not self.regress_forces:
            return energy, latent_rep
        else:
            return energy, forces, latent_rep
