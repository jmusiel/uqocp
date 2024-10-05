from ocpmodels.models.painn.painn import (
    PaiNN,
    registry,
    conditional_grad,
    torch,
    scatter,
)



@registry.register_model("latent_painn")
class latent_PaiNN(PaiNN):
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        z = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos = pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            edge_dist,
            edge_vector,
            id_swap,
        ) = self.generate_graph_values(data)

        assert z.dim() == 1 and z.dtype == torch.long

        edge_rbf = self.radial_basis(edge_dist)  # rbf * envelope

        x = self.atom_emb(z)
        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        #### Interaction blocks ###############################################

        for i in range(self.num_layers):
            dx, dvec = self.message_layers[i](
                x, vec, edge_index, edge_rbf, edge_vector
            )

            x = x + dx
            vec = vec + dvec
            x = x * self.inv_sqrt_2

            dx, dvec = self.update_layers[i](x, vec)

            x = x + dx
            vec = vec + dvec
            x = getattr(self, "upd_out_scalar_scale_%d" % i)(x)

        #### begin latent rep #####################################################
        latent_rep = x.detach()
        #### end latent rep #####################################################

        #### Output block #####################################################

        per_atom_energy = self.out_energy(x).squeeze(1)
        energy = scatter(per_atom_energy, batch, dim=0)

        if self.regress_forces:
            if self.direct_forces:
                forces = self.out_forces(x, vec)
                return energy, forces, latent_rep
            else:
                forces = (
                    -1
                    * torch.autograd.grad(
                        x,
                        pos,
                        grad_outputs=torch.ones_like(x),
                        create_graph=True,
                    )[0]
                )
                return energy, forces, latent_rep
        else:
            return energy, latent_rep