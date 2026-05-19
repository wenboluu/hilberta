import torch
import os
from pytorch_fid.fid_score import calculate_fid_given_paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORG_STATS_PATH = os.path.join(SCRIPT_DIR, "coco_test2017_stats.npz")


def compute_fid_against_org_stats(path_generated, org_stats_path, device):
    path_pair = [path_generated, org_stats_path]
    batch_size = 128
    dims = 2048

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    fid_value = calculate_fid_given_paths(
        path_pair, batch_size, device, dims, num_workers
    )
    return fid_value


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="Path to generated images folder")
    parser.add_argument("--stats", type=str, default=ORG_STATS_PATH, help="Path to reference stats npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing FID: {args.image_dir} vs {args.stats}")
    fid = compute_fid_against_org_stats(args.image_dir, args.stats, device)
    print(f"FID: {fid:.4f}")
