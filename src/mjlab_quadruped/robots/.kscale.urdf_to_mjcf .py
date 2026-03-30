"""
Convert a URDF to MJCF using urdf2mjcf with external metadata.
"""

import argparse
import json
from pathlib import Path

from urdf2mjcf import run
from urdf2mjcf.model import JointMetadata
from urdf2mjcf.model import ActuatorMetadata


def load_joint_metadata(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)["joint_name_to_metadata"]

    joint_metadata = {}
    for key, value in data.items():
        joint_metadata[key] = JointMetadata.from_dict(value)

    return joint_metadata


def load_actuator_metadata(path: Path) -> dict:
    with open(path, "r") as f:
        motor_data = json.load(f)

    actuator_type = motor_data["actuator_type"]
    actuator_metadata = {
        actuator_type: ActuatorMetadata.from_dict(motor_data)
    }

    return actuator_metadata


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="URDF to MJCF converter")

    parser.add_argument("--urdf", required=True, type=Path, help="Path to URDF file")
    parser.add_argument("--mjcf", required=True, type=Path, help="Output MJCF file")

    parser.add_argument(
        "--joint-metadata",
        required=True,
        type=Path,
        help="Path to joint_metadata.json",
    )

    parser.add_argument(
        "--actuator-metadata",
        required=True,
        type=Path,
        help="Path to actuator JSON file",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Path to metadata.json",
    )

    parser.add_argument(
        "--copy-meshes",
        action="store_true",
        help="Copy mesh files to MJCF directory",
    )

    args = parser.parse_args()

    urdf_path = args.urdf
    mjcf_path = args.mjcf

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    joint_metadata = load_joint_metadata(args.joint_metadata)
    actuator_metadata = load_actuator_metadata(args.actuator_metadata)

    run(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        copy_meshes=args.copy_meshes,
        metadata_file=args.metadata,
        joint_metadata=joint_metadata,
        actuator_metadata=actuator_metadata,
    )

    if not mjcf_path.exists():
        raise RuntimeError("Error during MJCF file creation")
