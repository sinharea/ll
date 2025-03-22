import numpy as np
import gymnasium as gym
from multiprocessing import Pool, cpu_count
import os


def policy_action(params, observation):
    W = params[:32].reshape(8, 4)
    b = params[32:36]
    logits = np.dot(observation, W) + b
    return np.argmax(logits)


def evaluate(params):
    env = gym.make("LunarLander-v3")
    total_reward = 0.0
    for _ in range(100):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = policy_action(params, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
    env.close()
    return total_reward / 100


def load_checkpoint():
    if os.path.exists('checkpoint.npz'):
        data = np.load('checkpoint.npz')
        return (
            data['mean'],
            data['var'],
            data['best_params'],
            float(data['best_average']),
            int(data['total_episodes'])
        )
    return None


def save_checkpoint(mean, var, best_params, best_average, total_episodes):
    np.savez('checkpoint.npz',
             mean=mean,
             var=var,
             best_params=best_params,
             best_average=best_average,
             total_episodes=total_episodes)


def train():
    # Hyperparameters
    # n_samples = 200
    # n_elite = 40
    # noise_scale = 0.1
    n_samples = 500
    n_elite = 100    
    noise_scale = 0.05


    # Try loading checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        mean, var, best_params, best_average, total_episodes = checkpoint
        print(f"\nResumed training with best average: {best_average:.2f}")
    else:
        # Initialize new training
        mean = np.random.randn(36) * 0.1
        var = np.ones(36) * noise_scale
        best_average = -np.inf
        best_params = None
        total_episodes = 0

    with Pool(processes=cpu_count()) as pool:
        while best_average < 310:
            # Generate samples
            samples = mean + np.random.randn(n_samples, 36) * np.sqrt(var)

            # Evaluate population
            avg_rewards = np.array(pool.map(evaluate, samples))
            current_best = np.max(avg_rewards)
            current_avg = np.mean(avg_rewards)
            total_episodes += n_samples * 1000

            # Update elite distribution
            elite_indices = np.argsort(avg_rewards)[-n_elite:]
            elites = samples[elite_indices]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0) + 1e-5

            # Update best policy
            if current_best > best_average:
                best_average = current_best
                best_params = samples[np.argmax(avg_rewards)]
                np.save('best_policy.npy', best_params)
                print(f"\nNew best average: {best_average:.2f} - Policy saved!")

            # Print progress
            print(f"Episodes: {total_episodes} | "
                  f"Overall Best: {best_average:.2f} | "
                  f"Current Best: {current_best:.2f} | "
                  f"Population Avg: {current_avg:.2f}")

            # Save checkpoint
            save_checkpoint(mean, var, best_params, best_average, total_episodes)


if __name__ == "__main__":
    train()
    # Cleanup checkpoint after successful training
    if os.path.exists('checkpoint.npz'):
        os.remove('checkpoint.npz')
