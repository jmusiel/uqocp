from ocpmodels.models.equiformer_v2.trainers.forces_trainer import EquiformerV2ForcesTrainer
from experimental.jmusiel.save_latent_rep.latent_inference_trainer import LatentTrainer
from ocpmodels.common.registry import registry

@registry.register_trainer("equiformerv2_latent")
class EquiformerV2LatentTrainer(EquiformerV2ForcesTrainer, LatentTrainer):
    """
    Trainer class for EquiformerV2 model with latent output.
    Combines latent trainer and equiformerv2 forces trainer, which each modify separate methods of the original ForcesTrainer.
    """
    pass