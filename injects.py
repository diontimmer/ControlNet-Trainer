from cldm.hack import disable_verbosity
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings(
    "ignore",
    ".*The dataloader, train_dataloader, does not have many workers which may be a bottleneck*",
)
warnings.filterwarnings(
    "ignore",
    ".*You defined a `validation_step` but have no `val_dataloader`. Skipping val loop*",
)
warnings.filterwarnings(
    "ignore",
    ".*in your `training_step` but the value needs to be floating point. Converting it to torch.float32*",
)

disable_verbosity()
