#!/usr/bin/env bash
# Re-apply the Isaac Lab 2.0.1 / rsl-rl 2.1.2 compatibility patches.
#
# The repo targets a newer Isaac Lab than the "boa" training PC has installed,
# so a handful of API names differ. These edits are MACHINE-LOCAL: never commit
# them (they break on newer Isaac Lab). Because they live in the working tree,
# every `git stash` / `git checkout` / `git pull` wipes them — so just run this
# after any git operation:
#
#     bash scripts/boa_compat.sh                # re-apply patches
#     bash scripts/boa_compat.sh plane          # ...and switch to flat ground
#     bash scripts/boa_compat.sh generator      # ...and switch to mixed terrain
#
# Safe to run repeatedly (each edit is a no-op once applied). Delete this script
# once the lab upgrades Isaac Lab to 2.1+ and rsl-rl to 3.0.1.
set -e
cd "$(dirname "$0")/.."   # -> eigenbot/eigenbot

ASSETS="source/eigenbot/eigenbot/assets/eigenbot.py"
ENVCFG="source/eigenbot/eigenbot/tasks/direct/eigenbot/eigenbot_env_cfg.py"

# 1. ImplicitActuatorCfg: effort_limit_sim (new) -> effort_limit (2.0)
sed -i 's/effort_limit_sim=/effort_limit=/' "$ASSETS"

# 2. RayCasterCfg: ray_alignment="yaw" (new) -> attach_yaw_only=True (2.0)
sed -i 's/ray_alignment="yaw",/attach_yaw_only=True,/' "$ENVCFG"

# 3. Relax the rsl-rl version gate in the training script
sed -i 's/RSL_RL_VERSION = "3.0.1"/RSL_RL_VERSION = "2.1.2"/' scripts/rsl_rl/train.py

# 4. train.py / play.py: drop rsl-rl 3.x-only APIs
python3 - << 'PYEOF'
for path in ["scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py"]:
    s = open(path).read()
    s = s.replace("from rsl_rl.runners import DistillationRunner, OnPolicyRunner",
                  "from rsl_rl.runners import OnPolicyRunner")
    s = s.replace("from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg",
                  "from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg as RslRlBaseRunnerCfg")
    s = s.replace("RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)",
                  "RslRlVecEnvWrapper(env)")
    s = s.replace('agent_cfg.algorithm.class_name == "Distillation"',
                  'getattr(agent_cfg.algorithm, "class_name", "PPO") == "Distillation"')
    s = s.replace('if agent_cfg.class_name == "OnPolicyRunner":',
                  'if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "OnPolicyRunner":')
    s = s.replace('elif agent_cfg.class_name == "DistillationRunner":', 'elif False:')
    s = s.replace('runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)',
                  'pass')
    s = s.replace('runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)',
                  'pass')
    s = s.replace("from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint\n", "")
    s = s.replace('resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)',
                  'resume_path = None')
    s = s.replace("    obs = env.get_observations()\n", "    obs, _ = env.get_observations()\n")
    open(path, "w").write(s)
PYEOF

# 5. rsl-rl 2.x RslRlPpoActorCriticCfg has no *_obs_normalization fields
#    (observation normalisation is the runner-level empirical_normalization flag,
#    already False in both configs, so behaviour is unchanged).
sed -i '/actor_obs_normalization/d;/critic_obs_normalization/d' \
    source/eigenbot/eigenbot/tasks/direct/eigenbot/agents/rsl_rl_ppo_cfg.py \
    source/eigenbot/eigenbot/tasks/direct/eigenbot/agents/rsl_rl_cpg_ppo_cfg.py

# 6. Optional terrain switch (flat ground for gait inspection, generator for training)
case "${1:-}" in
    plane)
        sed -i 's/terrain_type="generator"/terrain_type="plane"/' "$ENVCFG"
        echo "[terrain] flat plane"
        ;;
    generator)
        sed -i 's/terrain_type="plane"/terrain_type="generator"/' "$ENVCFG"
        echo "[terrain] mixed generator"
        ;;
esac

echo "[ok] compat patches applied:"
grep -n "effort_limit=" "$ASSETS" | head -1
grep -n "attach_yaw_only\|terrain_type" "$ENVCFG" | head -2
grep -n 'RSL_RL_VERSION' scripts/rsl_rl/train.py | head -1
