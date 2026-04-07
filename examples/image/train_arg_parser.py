import argparse
import logging

from models.model_configs import MODEL_CONFIGS

DATASET_CHOICES = [name for name in MODEL_CONFIGS if not name.endswith("_discrete")]

logger = logging.getLogger(__name__)

PATH_FAMILIES = ["linear", "trig_vp"]
CLOCK_FAMILIES = [
    "uniform",
    "ft_beta",
    "poly_a0.5",
    "poly_a2.0",
    "cosine",
    "sigmoid_k8",
    "exp_l3",
]
SAMPLING_SOLVERS = ["euler", "heun2", "rk3", "stork4"]
SUPPORTED_METRICS = ["fid", "precision_recall", "inception_score"]
MODEL_OUTPUT_TYPES = ["velocity", "base_velocity"]
TIME_SAMPLING_STRATEGIES = [
    "uniform",
    "ds_dr_sq",
    "mixed_lambda",
    "stratified",
    "stratified_mixed",
    "curriculum",
]
SHARED_CLOCK_MODES = ["off", "ge_stork"]
SHARED_CLOCK_FAMILIES = ["va", "vb", "aa", "ab"]
SHARED_CLOCK_PILOT_SOLVERS = ["euler", "heun2", "stork4"]
SHARED_CLOCK_JACOBIAN_BACKENDS = ["exact", "probe"]


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value}.")


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)",
    )
    parser.add_argument("--epochs", default=921, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument(
        "--optimizer_betas",
        nargs="+",
        type=float,
        default=[0.9, 0.95],
        help="Optimizer betas",
    )
    parser.add_argument(
        "--decay_lr",
        action="store_true",
        help="Adds a linear decay to the lr during training.",
    )
    parser.add_argument(
        "--class_drop_prob",
        type=float,
        default=1.0,
        help="Probability to drop conditioning during training",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="When evaluating, use the model exponential moving average weights.",
    )

    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=DATASET_CHOICES,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data_path",
        default="./data/image_generation",
        type=str,
        help="Dataset root path.",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="Path where to save checkpoints and logs.",
    )

    parser.add_argument(
        "--path_family",
        default="linear",
        choices=PATH_FAMILIES,
        help="Base probability path used for continuous flow matching.",
    )
    parser.add_argument(
        "--clock_family",
        default="uniform",
        choices=CLOCK_FAMILIES,
        help="Clock family used to reparameterize the base path.",
    )
    parser.add_argument(
        "--clock_beta",
        type=float,
        default=None,
        help="Beta parameter used by FT-clock families.",
    )
    parser.add_argument(
        "--sampling_solver",
        default="heun2",
        choices=SAMPLING_SOLVERS,
        help="Fixed-step solver used during continuous sampling.",
    )
    parser.add_argument(
        "--eval_nfe",
        default=50,
        type=int,
        help="Evaluation NFE budget counted as real network forward calls.",
    )
    parser.add_argument(
        "--shared_clock_mode",
        default="off",
        choices=SHARED_CLOCK_MODES,
        help=(
            "Offline shared clock branch. "
            "off keeps the existing runtime unchanged; ge_stork builds one shared clock "
            "offline and reuses it for all evaluation budgets."
        ),
    )
    parser.add_argument(
        "--shared_clock_family",
        default="ab",
        choices=SHARED_CLOCK_FAMILIES,
        help="Shared clock family: va, vb, aa, or ab.",
    )
    parser.add_argument(
        "--shared_clock_physical_grid_size",
        default=65,
        type=int,
        help="Number of physical-time nodes used by the offline pilot grid.",
    )
    parser.add_argument(
        "--shared_clock_pilot_solver",
        default="heun2",
        choices=SHARED_CLOCK_PILOT_SOLVERS,
        help="Baseline solver used to generate pilot trajectories on the physical grid.",
    )
    parser.add_argument(
        "--shared_clock_pilot_batch_size",
        default=16,
        type=int,
        help="Calibration batch size used when collecting pilot trajectories.",
    )
    parser.add_argument(
        "--shared_clock_pilot_num_batches",
        default=4,
        type=int,
        help="Number of calibration batches used to build the offline shared clock.",
    )
    parser.add_argument(
        "--shared_clock_observation_microbatch",
        default=4,
        type=int,
        help="Trajectory microbatch size used during shared clock local-object extraction.",
    )
    parser.add_argument(
        "--shared_clock_eps",
        default=1.0e-6,
        type=float,
        help="Numerical epsilon used by the shared clock amplitude proxies.",
    )
    parser.add_argument(
        "--shared_clock_jacobian_backend",
        default="probe",
        choices=SHARED_CLOCK_JACOBIAN_BACKENDS,
        help="Backend used for generator-layer Jacobian extraction.",
    )
    parser.add_argument(
        "--shared_clock_jacobian_num_probes",
        default=4,
        type=int,
        help="Number of Hutchinson probes used by the probe Jacobian backend.",
    )
    parser.add_argument(
        "--shared_clock_optimizer_steps",
        default=200,
        type=int,
        help="Number of Adam steps used by the V-b / A-b shared clock optimizer.",
    )
    parser.add_argument(
        "--shared_clock_optimizer_lr",
        default=0.05,
        type=float,
        help="Adam learning rate used by the V-b / A-b shared clock optimizer.",
    )
    parser.add_argument(
        "--shared_clock_cache_path",
        default="none",
        type=str,
        help="Optional cache file for the offline shared clock profile. Use 'none' to disable caching.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid"],
        choices=SUPPORTED_METRICS,
        help="Metrics to compute during evaluation.",
    )
    parser.add_argument(
        "--precision_recall_neighbors",
        default=3,
        type=int,
        help="Neighborhood size used by the precision/recall manifold metric.",
    )
    parser.add_argument(
        "--precision_recall_max_samples",
        default=10000,
        type=int,
        help="Maximum number of real and fake samples used for precision/recall.",
    )
    parser.add_argument(
        "--inception_score_splits",
        default=10,
        type=int,
        help="Number of splits used when computing inception score.",
    )
    parser.add_argument(
        "--analysis_num_bins",
        default=20,
        type=int,
        help="Number of time bins used by analysis scripts.",
    )
    parser.add_argument(
        "--analysis_num_batches",
        default=8,
        type=int,
        help="Number of data batches consumed by analysis scripts.",
    )
    parser.add_argument(
        "--analysis_num_samples",
        default=512,
        type=int,
        help="Number of synthetic samples used by analysis scripts.",
    )
    parser.add_argument(
        "--model_output_type",
        default="velocity",
        choices=MODEL_OUTPUT_TYPES,
        help="Whether the model predicts the true velocity or the base velocity.",
    )
    parser.add_argument(
        "--time_sampling_strategy",
        default="uniform",
        choices=TIME_SAMPLING_STRATEGIES,
        help="Training-time strategy for sampling the reparameterized time r.",
    )
    parser.add_argument(
        "--mixed_lambda",
        default=0.5,
        type=float,
        help="Mixture coefficient used by mixed_lambda and stratified_mixed strategies.",
    )
    parser.add_argument(
        "--stratified_bins",
        default=16,
        type=int,
        help="Number of equal-width bins used by stratified sampling strategies.",
    )

    parser.add_argument(
        "--sym",
        default=0.0,
        type=float,
        help="Symmetric term for sampling the discrete flow.",
    )
    parser.add_argument(
        "--temp",
        default=1.0,
        type=float,
        help="Temperature for sampling the discrete flow.",
    )
    parser.add_argument(
        "--sym_func",
        action="store_true",
        help="Use a fixed function for the symmetric term in the discrete flow.",
    )
    parser.add_argument(
        "--sampling_dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Solver dtype for sampling the discrete flow.",
    )
    parser.add_argument(
        "--cfg_scale",
        default=0.0,
        type=float,
        help="Classifier-free guidance scale for generating samples.",
    )
    parser.add_argument(
        "--fid_samples",
        default=50000,
        type=int,
        help="Number of synthetic samples for evaluation.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume from checkpoint")

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        metavar="N",
        help="Start epoch (used when resumed from checkpoint)",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="No training, only run evaluation"
    )
    parser.add_argument(
        "--eval_frequency",
        default=50,
        type=int,
        help="Frequency (in number of epochs) for running evaluation. -1 to never run evaluation.",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="Backward-compatible flag that forces FID computation during evaluation.",
    )
    parser.add_argument(
        "--save_fid_samples",
        action="store_true",
        help="Save all samples generated for FID computation.",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient transfer.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="URL used to set up distributed training"
    )
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Only run one batch of training and evaluation.",
    )
    parser.add_argument(
        "--discrete_flow_matching",
        action="store_true",
        help="Train discrete flow matching model.",
    )
    parser.add_argument(
        "--discrete_fm_steps",
        default=1024,
        type=int,
        help="Number of sampling steps for discrete FM.",
    )
    return parser
