import torch
import torch.nn.functional as F

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    mat = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return mat.reshape(quaternions.shape[:-1] + (3, 3))

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    subgradient is zero where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    

def convert_pos_quat_to_mat(obj_pose_pos_quat):

    pos = obj_pose_pos_quat[:, 0:3]
    quat_xyzw = obj_pose_pos_quat[:, 3:7]

    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]   
    R = quaternion_to_matrix(quat_wxyz)

    T = torch.eye(4).repeat(R.shape[0], 1, 1).to(R.device)
    T[:, 0:3, 0:3] = R
    T[:, 0:3, 3] = pos

    return T    

def compute_poses_wrt_root(object_pose, root_pose):
    assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
    object_pos = object_pose[:, 0:3]
    object_rot = object_pose[:, 3:7]

    root_pos = root_pose[:, 0:3]
    root_quat_xyzw = root_pose[:, 3:7]

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]   
    R_W_P = quaternion_to_matrix(root_quat_wxyz)

    T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
    T_W_P[:, 0:3, 0:3] = R_W_P
    T_W_P[:, 0:3, 3] = root_pos

    object_quat_xyzw = object_rot
    object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
    R_W_O = quaternion_to_matrix(object_quat_wxyz)

    T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
    T_W_O[:, 0:3, 0:3] = R_W_O
    T_W_O[:, 0:3, 3] = object_pos

    relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

    relative_translation = relative_pose[:, 0:3, 3]
    relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

    relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

    object_pos_wrt_root = relative_translation
    object_quat_wrt_root = relative_quat_xyzw

    object_pose_wrt_root = torch.cat((object_pos_wrt_root, object_quat_wrt_root), axis=-1)
    
    return object_pose_wrt_root

def compute_poses_wrt_world(object_pose, root_pose):
    assert object_pose.shape == root_pose.shape, f"Different shapes of object poses and root_poses !!! {object_pose.shape},{root_pose.shape}"
    object_pos = object_pose[:, 0:3]
    object_rot = object_pose[:, 3:7]

    root_pos = root_pose[:, 0:3]
    root_quat_xyzw = root_pose[:, 3:7]

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]   
    R_W_P = quaternion_to_matrix(root_quat_wxyz)

    T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
    T_W_P[:, 0:3, 0:3] = R_W_P
    T_W_P[:, 0:3, 3] = root_pos

    object_quat_xyzw = object_rot
    object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
    R_P_O = quaternion_to_matrix(object_quat_wxyz)

    T_P_O = torch.eye(4).repeat(R_P_O.shape[0], 1, 1).to(R_P_O.device)
    T_P_O[:, 0:3, 0:3] = R_P_O
    T_P_O[:, 0:3, 3] = object_pos

    pose = torch.matmul(T_W_P, T_P_O)

    translation = pose[:, 0:3, 3]
    quat_wxyz = matrix_to_quaternion(pose[:, 0:3, 0:3])

    relative_quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    object_pos_wrt_world = translation
    object_quat_wrt_world = relative_quat_xyzw

    object_pose_wrt_world = torch.cat((object_pos_wrt_world, object_quat_wrt_world), axis=-1)
    
    return object_pose_wrt_world