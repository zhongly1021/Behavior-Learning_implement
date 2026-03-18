"""Minimal runnable BL example (continuous regression)."""

import torch

from blnetwork import BLDeep, ContinuousTrainer, TrainConfig, OptimConfig


def make_toy_data(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (1.5 * x[:, :1] - 0.8 * x[:, 1:2] + 0.1 * torch.randn(n, 1))
    return x, y


def main() -> None:
    x_train, y_train = make_toy_data(512)
    x_val, y_val = make_toy_data(128)

    model = BLDeep(hidden_dims=[16, 8], task="continuous")
    trainer = ContinuousTrainer(
        model=model,
        optim_cfg=OptimConfig(lr=1e-3, weight_decay=1e-4),
        train_cfg=TrainConfig(max_epochs=20, batch_size=64, early_stop=False, verbose=False),
    )

    result = trainer.fit(x_train, y_train, x_val, y_val)

    with torch.no_grad():
        score = model(x_val[:5], y_val[:5])

    print("Training done.")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Last train loss: {result['history']['train_loss'][-1]:.6f}")
    print("Example score shape:", tuple(score.shape))


if __name__ == "__main__":
    main()
