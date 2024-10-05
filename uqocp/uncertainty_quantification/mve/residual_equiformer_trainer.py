from ocpmodels.models.equiformer_v2.trainers.forces_trainer import (
    EquiformerV2ForcesTrainer,
    distutils,
    DistributedDataParallel,
    OCPDataParallel,
    logging,
    
)
from uqocp.uncertainty_quantification.mve.residual_trainer import ResidualTrainer
from ocpmodels.common.registry import registry

@registry.register_trainer("equiformerv2_residual")
class EquiformerV2ResidualTrainer(EquiformerV2ForcesTrainer, ResidualTrainer):
    """
    Trainer class for EquiformerV2 model with residual output.
    Combines residual trainer and equiformerv2 forces trainer, which each modify separate methods of the original ForcesTrainer.
    """
    pass
    def load_model(self):
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']}")

        # TODO: depreicated, remove.
        bond_feat_dim = None
        bond_feat_dim = self.config["model_attributes"].get(
            "num_gaussians", 50
        )

        loader = self.train_loader or self.val_loader or self.test_loader
        self.model = registry.get_model_class(self.config["model"])(
            loader.dataset[0].x.shape[-1]
            if loader
            and hasattr(loader.dataset[0], "x")
            and loader.dataset[0].x is not None
            else None,
            bond_feat_dim,
            self.num_targets,
            **self.config["model_attributes"],
        ).to(self.device)

        # for no weight decay
        self.model_params_no_wd = {}
        if hasattr(self.model, "no_weight_decay"):
            self.model_params_no_wd = self.model.no_weight_decay()

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                find_unused_parameters=True,
            )