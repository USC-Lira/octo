"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb
from functools import partial
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

WANDB_PROJECT_NAME="jesse_finetune_octo"
WANDB_ENTITY_NAME="clvr"

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 16, "Batch size for finetuning.")
flags.DEFINE_integer("train_steps", 50000, "Training steps")
flags.DEFINE_integer("num_eval_batches", 16, "Num batches to perform each eval step")
flags.DEFINE_bool("train_proprio", False, "Whether to train proprio")
flags.DEFINE_string("wandb_project_name", WANDB_PROJECT_NAME, "Wandb project name")
flags.DEFINE_string("wandb_entity_name", WANDB_ENTITY_NAME, "Wandb entity name")
flags.DEFINE_string("wandb_run_name", None, "Wandb run name")

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)

def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(resume=FLAGS.wandb_run_name, project=FLAGS.wandb_project_name, entity=FLAGS.wandb_entity_name)

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    dataset_kwargs=dict(
        name="bridge_42_trajs",
        data_dir=FLAGS.data_dir,
        image_obs_keys={"primary": "image_1"},
        language_key="language_instruction",
    )
    if FLAGS.train_proprio:
        dataset_kwargs["proprio_obs_key"]="state"
    traj_transform_kwargs = dict(
        window_size=2,
        action_horizon=4,
    )
    frame_transform_kwargs = dict(
        resize_size={"primary": (256, 256)},
    )

    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )
    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    eval_dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
        train=False,
    )
    eval_data_iter = (
        eval_dataset.repeat()
        .unbatch()
        .shuffle(1000)
        .batch(FLAGS.batch_size // jax.device_count())
        .iterator()
    )
    eval_data_iter = map(process_batch, eval_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    if FLAGS.train_proprio:
        config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
            LowdimObsTokenizer,
            n_bins=256,
            bin_type="normal",
            low=-2.0,
            high=2.0,
            obs_keys=["proprio"],
        )
    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=4,
        action_dim=7,
        readout_key="readout_action",
    )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    # val_callback = ValidationCallback(
    #    loss_fn=loss_fn,
    #    process_batch_fn=process_batch,
    #    text_processor=text_processor,
    #    val_dataset_kwargs_list=[dataset_kwargs],
    #    dataset_kwargs={
    #        **dataset_kwargs,
    #        "frame_transform_kwargs": frame_transform_kwargs,
    #        "traj_transform_kwargs": traj_transform_kwargs,
    #        "batch_size": FLAGS.batch_size,
    #    },
    #    modes_to_evaluate=["text_conditioned"],
    #    **dict(
    #        val_shuffle_buffer_size=1000,
    #        num_val_batches=16,
    #    ),
    # )
    # Replicate the initial model state across devices
    train_state = jax.device_put_replicated(train_state, jax.devices())

    def inner_train_step(state, batch, train=True):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=train
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @partial(jax.pmap, axis_name="batch")
    def train_step(state, batch, train=True):
        return inner_train_step(state, batch, train=train)

    @jax.jit
    def single_device_eval_step(state, batch, train=False):
        return inner_train_step(state, batch, train=train)

    # Split the batch data across devices
    def shard_batch(batch):
        return jax.tree_map(
            lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), batch
        )

    # Run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(
        range(FLAGS.train_steps), total=FLAGS.train_steps, dynamic_ncols=True
    ):
        batch = next(train_data_iter)
        batch = shard_batch(batch)
        train_state, update_info = train_step(train_state, batch)
        eval_metrics = {}
        if (i + 1) % 10000 == 0:
            ## Eval model
            # logging.info("Evaluating model...")
            # Gather the replicated state from multiple devices
            # single_device_state = jax.tree_map(lambda x: x[0], train_state)

            # for i in range(FLAGS.num_eval_batches):
            #    eval_batch = next(eval_data_iter)
            #    _, eval_info = single_device_eval_step(
            #        single_device_state, eval_batch, train=False
            #    )
            #    eval_info = jax.device_get(eval_info)
            #    eval_metrics = (
            #        flax.traverse_util.flatten_dict({"validation": eval_info}, sep="/"),
            #    )

            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)

        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            train_metrics = flax.traverse_util.flatten_dict(
                {"training": update_info}, sep="/"
            )
            wandb.log(
                {**train_metrics, **eval_metrics},
                step=i,
            )

if __name__ == "__main__":
    app.run(main)
