# Anonymous Code for Graph-Grounded Process Supervision

This repository is the anonymous implementation accompanying a paper submission
on graph-grounded process supervision and future influence attribution for
reinforcement learning on graph reasoning.

The implementation is built on
[VERL](https://github.com/volcengine/verl). The upstream license and package
metadata are preserved. Training data and model checkpoints are not included
because of their size and distribution terms.

## Method Components

The main implementation is organized as follows:

- `verl/workers/reward_manager/graph_logic_observer.py`: response
  segmentation, graph grounding, task-state execution, verified milestones,
  and action-span extraction.
- `verl/workers/reward_manager/graph_progress_reward.py`: process reward
  composition, reward budgeting, and token-span mapping.
- `verl/trainer/ppo/core_algos.py`: group-relative episode advantage, step
  advantage construction, and future influence attribution.
- `verl/trainer/ppo/ray_trainer.py`: rollout, reward, advantage, and actor
  update integration.
- `verl/workers/actor/dp_actor.py`: policy-loss execution.
- `verl/workers/config/actor.py` and `verl/trainer/config/actor/actor.yaml`:
  method configuration.
- `scripts/run_training.sh`: one self-contained entry point for the final
  configuration.

The default script enables the paper configuration with:

- group-relative episode and action-level step advantages;
- graph-state transition verification;
- reliability-gated milestones for the selected task families;
- action-local future influence attribution;
- response-level positive reward budgeting; and
- outcome reward broadcast to generated tokens.

## Installation

Use Python 3.10 or later in an environment with a CUDA-compatible PyTorch
installation:

```bash
pip install -r requirements.txt
pip install -e .
```

Hardware and dependency versions may need to be adjusted for the local CUDA
driver. The code follows the standard VERL FSDP and vLLM training stack.

## Data Contract

The runner expects training and validation files in Parquet format. Each row
must follow the VERL prompt and reward schema used by the Erdos-style graph
tasks:

- `prompt`: chat-formatted model input;
- `reward_model.ground_truth`: final-answer target;
- `extra_info.task`: graph task identifier;
- `extra_info`: task metadata required by the corresponding verifier, such as
  graph edges, node sets, direction, source/target nodes, or weights.

The exact metadata fields vary by task and are consumed conservatively by
`graph_logic_observer.py`. Unsupported or incomplete metadata yields no
positive process credit.

## Training

Set local model and dataset paths through environment variables:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/test.parquet \
OUTPUT_DIR=./outputs/graphlogic_core4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/run_training.sh
```

Common resource settings can also be overridden without editing the script:

```bash
N_GPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
ROLLOUT_N=8 \
TOTAL_TRAINING_STEPS=43 \
bash scripts/run_training.sh
```

The script logs to `OUTPUT_DIR/training.log` and writes checkpoints according
to `SAVE_FREQ`.

## Anonymous Artifact Checks

Before publishing a revision, run:

```bash
bash scripts/check_anonymity.sh
```

The check rejects common private paths, private network addresses, experiment
artifacts, and likely embedded credentials. It is intentionally conservative;
review every match before uploading.

## Scope

This artifact contains the implementation required to inspect and run the
method. It intentionally excludes:

- datasets and generated trajectories;
- pretrained and post-trained checkpoints;
- evaluation outputs and training logs;
- paper sources and figures;
- cluster launch wrappers and machine-specific configuration; and
- local experiment history.

# GraphSCA
