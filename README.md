# Eigenbot IsaacLab
Repository for Eigenbot IsaacLab simulation and RL code, ported from the Eigenbot IsaacGym implementation.

## Environment Setup
The following environment setup steps has been tested on a Linux Ubuntu 20.04.6 LTS system. For newer Linux or Windows systems, alternative IsaacLab installation methods (using pip or pre-installed binary methods) can be used.

Docker containerization bypasses versioning issues, so this is the method described below. Ensure that Docker, Docker Compose, and Nvidia Container Toolkit is installed.

1. Clone the GitHub repository
```
git clone https://github.com/biorobotics/eigenbot_isaaclab
cd eigenbot_isaaclab
```

2. Build the Docker container. This may take a while (10-15 minutes). The `--files` flag adds an overlay that bind-mounts the eigenbot extension into the container.
```
python isaaclab/docker/container.py start --files eigenbot.yaml
```

3. Enter the Docker container. This will serve as the main development environment.
```
python isaaclab/docker/container.py enter --files eigenbot.yaml
```


## Training and Simulation
4. Inside the container, install the eigenbot extension and run it.
```bash
# Install the eigenbot extension (one-time, or after rebuilding the container)
cd /workspace/eigenbot
pip install -e source/eigenbot

# Render the eigenbot with zero actions
python scripts/zero_agent.py --task Template-Eigenbot-Direct-v0 --num_envs 1

# Move the eigenbot with random actions (untested)
python scripts/random_agent.py --task Template-Eigenbot-Direct-v0 --num_envs 1

# Train with PPO (no viz)
python scripts/rsl_rl/train.py --task Template-Eigenbot-Direct-v0 --num_envs 4096 --headless

# Train with visualization
python scripts/rsl_rl/train.py --task Template-Eigenbot-Direct-v0 --num_envs 4096

# Replay trained checkpoint
python scripts/rsl_rl/play.py --task Template-Eigenbot-Direct-v0 --num_envs 32
```

### Monitoring Training
During training, `rsl_rl` uses `tensorboard` as a visualizer to monitor training statistics such as loss and reward. You must first enter the docker image on a SEPARATE TERMINAL, then start the webserver to view these visualizations.

```bash
# Get the name of the docker container running the training job
docker ps

# Enter the docker image
docker exec -it <container_name> bash

# Inside the docker image
tensorboard --logdir /workspace/eigenbot/logs/rsl_rl/eigenbot_locomotion --port 6006
```

Then open the URL in a browser to view the visualizer.