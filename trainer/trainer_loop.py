"""
Minimal PPO-style training loop skeleton.
Consumes experience blocks from the replay buffer, updates a policy net,
and saves new checkpoints / policy_version.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from trainer.checkpointing import save_checkpoint
from trainer.models import MLPPolicy, make_optimizer
from trainer.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Trainer loop (skeleton).")
    parser.add_argument("--obs-dim", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-buffer", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--policy-version", type=int, default=0)
    parser.add_argument(
        "--base-llm-checkpoint",
        default=None,
        help="Optional path/name of a base LLM checkpoint to finetune (not used in stub MLP).",
    )
    parser.add_argument(
        "--pull-blocks-endpoint",
        default=None,
        help="If set, periodically poll this endpoint (e.g., trainer /api/experience_block feed) to pull experience blocks.",
    )
    return parser.parse_args()


def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(
    model: MLPPolicy,
    optimizer: torch.optim.Optimizer,
    batch: List[Tuple],
    args,
):
    obs, actions, rewards, dones, _, _ = zip(*batch)
    obs = torch.tensor(obs, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    rewards = list(rewards)
    dones = list(dones)

    logits, values = model(obs)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs[range(len(actions)), actions]

    # Dummy old_log_probs and values for skeleton; in practice, store them
    old_log_probs = action_log_probs.detach()
    old_values = values.detach()

    advantages, returns = compute_gae(rewards, old_values.tolist(), dones, args.gamma, args.gae_lambda)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(action_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)
    entropy_loss = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()

    loss = policy_loss + 0.5 * value_loss - args.entropy_coef * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_loss.item(),
    }


def main():
    args = parse_args()
    buffer = ReplayBuffer(capacity=args.buffer_capacity)
    model = MLPPolicy(args.obs_dim, args.action_dim)
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    policy_version = args.policy_version

    # TODO: if args.pull_blocks_endpoint is set, poll for new blocks and add to buffer.

    while True:
        if len(buffer) < args.min_buffer:
            # In a real system, sleep/poll for more data
            break

        batch = buffer.sample(args.batch_size)
        metrics = ppo_update(model, optimizer, batch, args)
        print(f"Policy version {policy_version} metrics: {metrics}")

        policy_version += 1
        save_checkpoint(model, optimizer, policy_version, args.checkpoint_dir)
        # TODO: notify policy service / update registry
        break  # stop after one iteration in skeleton


if __name__ == "__main__":
    main()
