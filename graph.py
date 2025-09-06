import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv('save_stat/td3_per_stat.csv')

# Set style for beautiful plots
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Calculate rolling averages for smoother curves
window = 200
df['avg_reward'] = df['score'].rolling(window).mean()
df['avg_steps'] = df['steps'].rolling(window).mean()
df['avg_y'] = df['y_position'].rolling(window).mean()
df['avg_actor_loss'] = df['actor_loss'].rolling(window).mean()
df['avg_critic_loss'] = df['critic_loss'].rolling(window).mean()

fig, axs = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('TD3+PER Training Progress', fontsize=18, fontweight='bold')

# Reward
sns.lineplot(x=df['episode'], y=df['score'], ax=axs[0, 0], label='Reward', color='tab:blue')
sns.lineplot(x=df['episode'], y=df['avg_reward'], ax=axs[0, 0], label='Avg Reward', color='tab:orange')
axs[0, 0].set_title('Reward per Episode')
axs[0, 0].legend()

# Steps
sns.lineplot(x=df['episode'], y=df['steps'], ax=axs[0, 1], label='Steps', color='tab:green')
sns.lineplot(x=df['episode'], y=df['avg_steps'], ax=axs[0, 1], label='Avg Steps', color='tab:red')
axs[0, 1].set_title('Steps per Episode')
axs[0, 1].legend()

# Y Position
sns.lineplot(x=df['episode'], y=df['y_position'], ax=axs[1, 0], label='Y Position', color='tab:purple')
sns.lineplot(x=df['episode'], y=df['avg_y'], ax=axs[1, 0], label='Avg Y', color='tab:pink')
axs[1, 0].set_title('Final Y Position per Episode')
axs[1, 0].legend()

# Actor Loss
sns.lineplot(x=df['episode'], y=df['actor_loss'], ax=axs[1, 1], label='Actor Loss', color='tab:olive')
sns.lineplot(x=df['episode'], y=df['avg_actor_loss'], ax=axs[1, 1], label='Avg Actor Loss', color='tab:cyan')
axs[1, 1].set_title('Actor Loss per Episode')
axs[1, 1].legend()

# Critic Loss
sns.lineplot(x=df['episode'], y=df['critic_loss'], ax=axs[2, 0], label='Critic Loss', color='tab:brown')
sns.lineplot(x=df['episode'], y=df['avg_critic_loss'], ax=axs[2, 0], label='Avg Critic Loss', color='tab:gray')
axs[2, 0].set_title('Critic Loss per Episode')
axs[2, 0].legend()

# Hide last subplot
axs[2, 1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()