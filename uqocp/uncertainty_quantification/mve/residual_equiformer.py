from typing import List, Optional
from ocpmodels.models.equiformer_v2.equiformer_v2_oc20 import (
    EquiformerV2_OC20,
    registry,
    conditional_grad,
    torch,
    SO3_Embedding,
    _AVG_NUM_NODES,
    FeedForwardNetwork,
)


@registry.register_model("residual_equiformer_v2")
class residual_EquiformerV2(EquiformerV2_OC20):
    """
    EquiformerV2 model with residual output
    """

    def __init__(
        self,
        num_atoms: int,
        bond_feat_dim: int,
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5,
        max_num_elements: int = 90,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: List[int] = ...,
        mmax_list: List[int] = ...,
        grid_resolution: Optional[int] = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0,
        weight_init: str = "normal",
    ):
        super().__init__(
            num_atoms,
            bond_feat_dim,
            num_targets,
            use_pbc,
            regress_forces,
            otf_graph,
            max_neighbors,
            max_radius,
            max_num_elements,
            num_layers,
            sphere_channels,
            attn_hidden_channels,
            num_heads,
            attn_alpha_channels,
            attn_value_channels,
            ffn_hidden_channels,
            norm_type,
            lmax_list,
            mmax_list,
            grid_resolution,
            num_sphere_samples,
            edge_channels,
            use_atom_edge_embedding,
            share_atom_edge_embedding,
            use_m_share_rad,
            distance_function,
            num_distance_basis,
            attn_activation,
            use_s2_act_attn,
            use_attn_renorm,
            ffn_activation,
            use_gate_act,
            use_grid_mlp,
            use_sep_s2_act,
            alpha_drop,
            drop_path_rate,
            proj_drop,
            weight_init,
        )
        self.residual_energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

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
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(
            len(data.natoms),
            device=node_energy.device,
            dtype=node_energy.dtype,
        )
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / _AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)

        ###############################################################
        # Residual (Energy) estimation
        ###############################################################
        residual_node_energy = self.residual_energy_block(x)
        residual_node_energy = residual_node_energy.embedding.narrow(1, 0, 1)
        residual_energy = torch.zeros(
            len(data.natoms),
            device=residual_node_energy.device,
            dtype=residual_node_energy.dtype,
        )
        residual_energy.index_add_(0, data.batch, residual_node_energy.view(-1))
        residual_energy = residual_energy / _AVG_NUM_NODES

        if not self.regress_forces:
            return energy, residual_energy
        else:
            return energy, forces, residual_energy
