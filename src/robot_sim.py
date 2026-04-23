"""
src/robot_sim.py
================
ロボットシミュレーション用のユーティリティ関数モジュール。
- move_arm  : 指定した FRAME にエンドエフェクタを移動させる（1ステップ分の制御）
- draw_axes : 指定した FRAME の座標軸を MuJoCo Mocap ボディで可視化する
- get_ee_frame : 現在のエンドエフェクタ姿勢を FRAME として取得
- mat2quat    : 3x3 回転行列を MuJoCo wxyz クォータニオンに変換
"""

from __future__ import annotations
from pathlib import Path
import sys

import mujoco
import numpy as np

# プロジェクトルートを sys.path に追加して geo をインポート
_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))
import geo

# --------------------------------------------------------------------------- #
#  内部ユーティリティ
# --------------------------------------------------------------------------- #

def mat2quat(mat: np.ndarray) -> np.ndarray:
    """3x3 回転行列を MuJoCo の wxyz クォータニオンに変換する。"""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def get_ee_frame(model: mujoco.MjModel, data: mujoco.MjData,
                 site_name: str = "attachment_site") -> geo.FRAME:
    """現在のエンドエフェクタ姿勢を geo.FRAME として返す。
    デフォルトで X軸周りに180度回転させた状態で返す。

    Parameters
    ----------
    model, data : MuJoCo モデルとデータ
    site_name   : エンドエフェクタを表す site の名前

    Returns
    -------
    geo.FRAME : 現在の手先座標系
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mat = data.site_xmat[site_id].reshape(3, 3).tolist()
    vec = data.site_xpos[site_id].tolist()
    # サイトの生フレーム
    f_site = geo.FRAME(mat=mat, vec=vec)
    # X軸周りに180度回転 (Roll=math.pi) したオフセットを適用
    import math
    f_offset = geo.FRAME(xyzrpy=[0, 0, 0, math.pi, 0, 0])
    return f_site * f_offset


# --------------------------------------------------------------------------- #
#  メインAPI
# --------------------------------------------------------------------------- #

def move_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_frame: geo.FRAME,
    *,
    site_name: str = "attachment_site",
    gain: float = 0.5,
    jacp: np.ndarray | None = None,
    jacr: np.ndarray | None = None,
) -> None:
    """ヤコビ行列の擬似逆行列を使って、手先を target_frame に近づける（1ステップ）。

    Parameters
    ----------
    model, data    : MuJoCo モデルとデータ
    target_frame   : 目標座標系 (geo.FRAME インスタンス)
    site_name      : エンドエフェクタ site の名前
    gain           : IK ゲイン（大きいほど速く収束するが不安定になりやすい）
    jacp, jacr     : 事前確保済みのヤコビ行列バッファ（省略時は内部で確保）

    注意
    ----
    この関数は data.ctrl を更新するだけで mj_step は呼ばない。
    呼び出し側で mujoco.mj_step(model, data) を実行してください。
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    if jacp is None:
        jacp = np.zeros((3, model.nv))
    if jacr is None:
        jacr = np.zeros((3, model.nv))

    # ---- 1. 現在の手先姿勢を取得 ----
    current_frame = get_ee_frame(model, data, site_name)
    T_current = current_frame.toarray()
    pos_current = T_current[0:3, 3]
    mat_current = T_current[0:3, 0:3]

    # ---- 2. 目標の位置・回転行列を取得 ----
    T_target = target_frame.toarray()
    pos_target = T_target[0:3, 3]
    mat_target = T_target[0:3, 0:3]

    # ---- 3. 誤差を計算（並進 + 回転） ----
    err_pos = pos_target - pos_current

    # 回転誤差: 各列ベクトルの外積の和（小角度近似による角速度ベクトル）
    err_rot = 0.5 * (
        np.cross(mat_current[:, 0], mat_target[:, 0]) +
        np.cross(mat_current[:, 1], mat_target[:, 1]) +
        np.cross(mat_current[:, 2], mat_target[:, 2])
    )

    err = np.hstack([err_pos, err_rot])  # 6次元誤差ベクトル

    # ---- 4. ヤコビ行列 → 擬似逆行列 → 関節速度 ----
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])           # (6, nv) ヤコビ行列
    dq = np.linalg.pinv(J) @ err * gain  # 関節速度ベクトル

    # ---- 5. 制御値を更新（積分: ctrl += dq * dt） ----
    for i in range(model.nu):
        data.ctrl[i] += dq[i] * model.opt.timestep



# --------------------------------------------------------------------------- #
#  ランタイム描画 (viewer.user_scn を使った動的ジオメトリ)
# --------------------------------------------------------------------------- #

def _dir_to_mat(d: np.ndarray) -> np.ndarray:
    """方向ベクトル d (単位ベクトル) に向く回転行列を返す。
    MuJoCo の mjv_initGeom が期待する行列: ローカル Z 軸がワールドの d を向く。

    Returns: shape=(9,) の row-major 3x3 回転行列
    """
    z = d / (np.linalg.norm(d) + 1e-12)
    # z に垂直な適当な x を決める
    ref = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    # 列ベクトル [x | y | z] → row-major flatten
    return np.column_stack([x, y, z]).flatten()


def _add_geom(
    scn: mujoco.MjvScene,
    geom_type: int,
    size: np.ndarray,
    pos: np.ndarray,
    mat: np.ndarray,
    rgba: tuple,
) -> None:
    """scn.geoms に 1 つジオメトリを追加する（容量超えは黙って無視）。"""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        geom_type,
        np.asarray(size,  dtype=np.float64),
        np.asarray(pos,   dtype=np.float64),
        np.asarray(mat,   dtype=np.float64),
        np.asarray(rgba,  dtype=np.float32),
    )
    scn.ngeom += 1


def clear_user_scene(viewer) -> None:
    """フレーム開始時に呼び出してユーザー描画ジオメトリをリセットする。"""
    viewer.user_scn.ngeom = 0


def draw_axes(
    viewer,
    frame: "geo.FRAME",
    *,
    origin_rgba: tuple = (1.0, 1.0, 0.0, 1.0),
    axis_radius: float = 0.005,
    axis_length: float = 0.1,
) -> None:
    """FRAME の座標系を viewer.user_scn にリアルタイム描画する。

    XML に mocap ボディを事前定義する必要がなく、任意の FRAME を渡すだけで
    座標軸（X: 赤, Y: 緑, Z: 青）と原点（球）を表示できる。

    Parameters
    ----------
    viewer       : mujoco.viewer.launch_passive() が返すビューアハンドル
    frame        : 描画したい座標系 (geo.FRAME インスタンス)
    origin_rgba  : 原点マーカー（球）の色 RGBA (0〜1)
    axis_radius  : 各軸の円柱の半径 [m]
    axis_length  : 各軸の長さ [m]
    """
    scn = viewer.user_scn
    T   = frame.toarray()
    origin = T[0:3, 3]
    R      = T[0:3, 0:3]  # 列ベクトルが各軸の世界座標方向

    eye = np.eye(3).flatten()

    # ---- 原点マーカー（球） ----
    r_sphere = axis_radius * 2
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE,
              [r_sphere, r_sphere, r_sphere],
              origin, eye, origin_rgba)

    # ---- 各軸のシリンダー ----
    half = axis_length / 2.0
    axis_colors = [
        (1.0, 0.1, 0.1, 1.0),  # X: 赤
        (0.1, 1.0, 0.1, 1.0),  # Y: 緑
        (0.1, 0.1, 1.0, 1.0),  # Z: 青
    ]
    for i, rgba in enumerate(axis_colors):
        d      = R[:, i]                    # 軸方向（ワールド座標）
        center = origin + half * d          # シリンダーの中心
        mat    = _dir_to_mat(d)             # ローカル Z → d になる回転行列
        size   = np.array([axis_radius, half, 0.0])
        _add_geom(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, size, center, mat, rgba)


def draw_box(
    viewer,
    frame: "geo.FRAME",
    *,
    size: tuple[float, float, float] = (0.02, 0.02, 0.02),
    rgba: tuple[float, float, float, float] = (0.9, 0.3, 0.1, 1.0),
) -> None:
    """FRAME の位置に箱ジオメトリを user_scn に描画する。

    Parameters
    ----------
    viewer  : mujoco viewer ハンドル
    frame   : 描画位置・姿勢 (geo.FRAME)
    size    : 箱の半辺長 [m] (x, y, z)
    rgba    : 箱の色 RGBA (0〜1)
    """
    scn = viewer.user_scn
    T = frame.toarray()
    origin = T[0:3, 3]
    R = T[0:3, 0:3]
    mat = R.flatten()
    _add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX,
              np.asarray(size), origin, mat, rgba)

