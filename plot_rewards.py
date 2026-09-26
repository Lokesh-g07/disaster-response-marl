import matplotlib.pyplot as plt
import re
import os

def parse_log(filename):
    episodes = []
    rewards = []
    
    if not os.path.exists(filename):
        return episodes, rewards

    with open(filename, 'r') as f:
        for line in f:
            if "Episode:" in line:
                try:
                    ep_match = re.search(r'Episode: (\d+)/\d+', line)
                    rew_match = re.search(r'Avg (?:Total )?Reward: ([-\d\.]+)', line)
                    if ep_match and rew_match:
                        episodes.append(int(ep_match.group(1)))
                        rewards.append(float(rew_match.group(1)))
                except:
                    pass
    return episodes, rewards

mappo_eps, mappo_rews = parse_log("mappo_train.log")
dqn_eps, dqn_rews = parse_log("dqn_train.log")

plt.figure(figsize=(10, 6))

if mappo_eps:
    plt.plot(mappo_eps, mappo_rews, label="MAPPO (CTDE Dual-Branch)", linewidth=2, color='tab:blue', marker='o', markersize=4)

if dqn_eps:
    plt.plot(dqn_eps, dqn_rews, label="Shared DQN (Dual-Branch)", linewidth=2, color='tab:orange', marker='s', markersize=4)

# Draw baseline targets for reference
plt.axhline(y=6.74, color='tab:green', linestyle='--', label="Greedy-Largest Zone (Oracle Baseline)")
plt.axhline(y=-14.31, color='tab:red', linestyle=':', label="Greedy-Nearest (Oracle Baseline)")

plt.title("AI Policy Convergence: Episode Reward over 350 Episodes")
plt.xlabel("Training Episodes")
plt.ylabel("Average Total Reward")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = r"C:\Users\prasa\.gemini\antigravity-ide\brain\b6ff76a5-2d51-45b2-8f85-9ac5d860b739\reward_plot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
