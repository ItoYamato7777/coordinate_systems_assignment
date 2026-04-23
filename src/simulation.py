"""
src/simulation.py
=================
ロボットシミュレーション実行クラス。

MuJoCo の初期化・メインループ・ウェイポイント切り替えを隠蔽し、
sample.py には「FRAME の定義・連結・draw_axes/move_arm の宣言」だけを
残すためのラッパーです。

使い方 (sample.py 側):
    sim = RobotSimulation(xml_path)
    sim.draw_axes(T_base2target_1, T_base2target_2)   # 描画したい FRAME を列挙
    sim.move_arm(T_base2target_1)                # ウェイポイントを追加
    sim.move_arm(T_base2target_2)
    sim.run()

ピッキング使い方:
    sim.spawn_object(T_obj_pos)             # オブジェクトを配置
    sim.move_arm(T_pick)                    # ピック位置へ移動
    sim.catch()                             # 到着したらオブジェクトを掴む
    sim.move_arm(T_place)                   # プレース位置へ移動
    sim.release()                           # 到着したらオブジェクトを離す
    sim.run()
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import mujoco
import mujoco.viewer
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))
import geo

from src.robot_sim import move_arm as _move_arm, draw_axes as _draw_axes
from src.robot_sim import draw_box as _draw_box
from src.robot_sim import get_ee_frame, clear_user_scene
from src.waypoint_manager import WaypointManager


# 座標軸の原点マーカー自動カラーパレット（登録順に割り当て）
_ORIGIN_PALETTE: list[tuple[float, float, float, float]] = [
    (1.0, 1.0, 0.0, 1.0),  # 黄
    (0.0, 1.0, 1.0, 1.0),  # 水色
    (0.5, 0.0, 1.0, 1.0),  # 紫
    (0.0, 1.0, 0.5, 1.0),  # 緑青
    (1.0, 0.0, 0.5, 1.0),  # ローズ
    (0.5, 1.0, 0.0, 1.0),  # 黄緑
]
_EE_ORIGIN_RGBA = (1.0, 0.5, 0.0, 1.0)  # 手先マーカー: 橙（固定）


class RobotSimulation:
    """MuJoCo ロボットシミュレーションを管理するクラス。

    sample.py では FRAME の定義・連結と、draw_axes / move_arm の
    宣言のみを記述すればよい。ループや IK 制御はすべてこのクラスが担う。

    Parameters
    ----------
    xml_path : str | Path
        MuJoCo シーン XML のパス
    site_name : str
        エンドエフェクタ site の名前
    ik_gain : float
        IK ゲイン（大きいほど速いが不安定）
    switch_threshold : float
        ウェイポイント切り替えの位置誤差閾値 [m]
    """

    def __init__(
        self,
        xml_path: str | Path,
        *,
        site_name: str = "attachment_site",
        ik_gain: float = 0.5,
        switch_threshold: float = 0.01,
    ) -> None:
        self._xml_path = Path(xml_path)
        self._site_name = site_name
        self._ik_gain = ik_gain
        self._switch_threshold = switch_threshold

        # ビルダー状態
        self._draw_entries: list[tuple[geo.FRAME, tuple]] = []  # (frame, rgba)
        self._waypoints: list[tuple[geo.FRAME, Callable[[], None] | None]] = []
        self._static_boxes: list[tuple[geo.FRAME, tuple, tuple]] = []  # (frame, size, rgba)

        # ピッキングオブジェクト
        self._obj_initial_frame: geo.FRAME | None = None
        self._obj_size: tuple[float, float, float] = (0.02, 0.02, 0.02)
        self._obj_rgba: tuple[float, float, float, float] = (0.9, 0.3, 0.1, 1.0)

        # ランタイム状態（run() 内で更新）
        self._obj_attached: bool = False      # 手先に追従中か
        self._obj_fixed_frame: geo.FRAME | None = None  # release 後の固定位置

    # ------------------------------------------------------------------
    #  Builder API
    # ------------------------------------------------------------------

    def draw_axes(self, *frames: geo.FRAME) -> "RobotSimulation":
        """毎フレーム座標軸を描画する FRAME を登録する。

        複数の FRAME を一度に渡すことができ、原点マーカーの色は
        登録順にパレットから自動付与される。

        Parameters
        ----------
        *frames : geo.FRAME
            描画する座標系（可変長）

        Returns
        -------
        RobotSimulation : メソッドチェーン用に self を返す
        """
        for frame in frames:
            idx = len(self._draw_entries)
            rgba = _ORIGIN_PALETTE[idx % len(_ORIGIN_PALETTE)]
            self._draw_entries.append((frame, rgba))
        return self

    def move_arm(self, frame: geo.FRAME) -> "RobotSimulation":
        """IK で追跡するウェイポイントを追加する（呼び出し順に追跡）。

        Parameters
        ----------
        frame : geo.FRAME
            目標座標系

        Returns
        -------
        RobotSimulation : メソッドチェーン用に self を返す
        """
        self._waypoints.append((frame, None))
        return self

    def add_box(
        self,
        frame: geo.FRAME,
        *,
        size: tuple[float, float, float] = (0.02, 0.02, 0.02),
        rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    ) -> "RobotSimulation":
        """静的な箱オブジェクト（パレットなど）をシーンに追加する。

        Parameters
        ----------
        frame : geo.FRAME
            箱の位置・姿勢
        size  : 箱の半辺長 [m] (x, y, z)
        rgba  : 箱の色 RGBA (0〜1)

        Returns
        -------
        RobotSimulation
        """
        self._static_boxes.append((frame, size, rgba))
        return self

    def spawn_object(
        self,
        frame: geo.FRAME,
        *,
        size: tuple[float, float, float] = (0.02, 0.02, 0.02),
        rgba: tuple[float, float, float, float] = (0.9, 0.3, 0.1, 1.0),
    ) -> "RobotSimulation":
        """ピッキング対象オブジェクトを配置する。

        物理ボディではなく、ランタイム描画（user_scn）で箱を表示する。
        catch() で手先に追従、release() でその場に固定される。

        Parameters
        ----------
        frame : geo.FRAME
            オブジェクトの初期位置・姿勢
        size  : 箱の半辺長 [m] (x, y, z)
        rgba  : 箱の色 RGBA (0〜1)

        Returns
        -------
        RobotSimulation
        """
        self._obj_initial_frame = frame
        self._obj_size = size
        self._obj_rgba = rgba
        return self

    def catch(self) -> "RobotSimulation":
        """直前のウェイポイント到達時にオブジェクトを掴む（手先に追従開始）。

        move_arm() の直後に呼ぶことで、そのウェイポイント到達時に
        オブジェクトがロボット手先にくっつくようになる。

        Returns
        -------
        RobotSimulation
        """
        if not self._waypoints:
            raise RuntimeError("catch() は move_arm() の後に呼んでください。")

        def _do_catch():
            self._obj_attached = True
            self._obj_fixed_frame = None
            print("[RobotSimulation] オブジェクトをキャッチしました")

        # 直前のウェイポイントにアクションを紐付け
        frame, _ = self._waypoints[-1]
        self._waypoints[-1] = (frame, _do_catch)
        return self

    def release(self) -> "RobotSimulation":
        """直前のウェイポイント到達時にオブジェクトを離す（その場に固定）。

        move_arm() の直後に呼ぶことで、そのウェイポイント到達時に
        オブジェクトがその位置に固定される。

        Returns
        -------
        RobotSimulation
        """
        if not self._waypoints:
            raise RuntimeError("release() は move_arm() の後に呼んでください。")

        def _do_release():
            self._obj_attached = False
            # 現在の手先位置を固定位置として記録
            # （この値は run() 内で ee_frame から設定）
            self._obj_pending_release = True
            print("[RobotSimulation] オブジェクトをリリースしました")

        frame, _ = self._waypoints[-1]
        self._waypoints[-1] = (frame, _do_release)
        self._obj_pending_release = False
        return self

    # ------------------------------------------------------------------
    #  Simulation loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """MuJoCo シミュレーションを起動してメインループを実行する。

        draw_axes / move_arm で登録した内容を毎ステップ実行する。
        ウィンドウが閉じられるまでループを継続する。
        """
        model = mujoco.MjModel.from_xml_path(str(self._xml_path))
        data  = mujoco.MjData(model)

        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        # 初期姿勢を制御値に反映（急な動きを防ぐ）
        for i in range(model.nu):
            data.ctrl[i] = data.qpos[i]

        # ヤコビ行列バッファを事前確保
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        # ランタイム状態を初期化
        self._obj_attached = False
        self._obj_fixed_frame = None
        self._obj_pending_release = False

        waypoint_mgr = WaypointManager(
            self._waypoints, threshold=self._switch_threshold
        ) if self._waypoints else None

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                self._on_step(model, data, viewer, waypoint_mgr, jacp, jacr)
                mujoco.mj_step(model, data)
                viewer.sync()

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _on_step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer: Any,
        waypoint_mgr: WaypointManager | None,
        jacp: np.ndarray,
        jacr: np.ndarray,
    ) -> None:
        """1 シミュレーションステップの処理。"""

        # 手先フレームを取得
        ee_frame = get_ee_frame(model, data, self._site_name)

        # ユーザー描画バッファをリセット
        clear_user_scene(viewer)

        # 登録された FRAME の座標軸を描画
        for frame, rgba in self._draw_entries:
            _draw_axes(viewer, frame, origin_rgba=rgba)

        # 手先座標軸（橙: 固定色）
        _draw_axes(viewer, ee_frame, origin_rgba=_EE_ORIGIN_RGBA)

        # 静的な箱を描画
        for f, s, r in self._static_boxes:
            _draw_box(viewer, f, size=s, rgba=r)

        # オブジェクト描画
        self._draw_object(viewer, ee_frame)

        # IK 制御 & ウェイポイント切り替え
        if waypoint_mgr is not None:
            _move_arm(
                model, data, waypoint_mgr.current,
                site_name=self._site_name,
                gain=self._ik_gain,
                jacp=jacp,
                jacr=jacr,
            )
            waypoint_mgr.update(ee_frame)

            # pending release の処理（release アクション後に手先位置を記録）
            if self._obj_pending_release:
                self._obj_fixed_frame = geo.FRAME(frm=ee_frame)
                self._obj_pending_release = False

    def _draw_object(self, viewer: Any, ee_frame: geo.FRAME) -> None:
        """ピッキングオブジェクトを描画する。"""
        if self._obj_initial_frame is None:
            return

        if self._obj_attached:
            # 手先に追従: 手先位置にオブジェクトを描画
            _draw_box(viewer, ee_frame,
                      size=self._obj_size, rgba=self._obj_rgba)
        elif self._obj_fixed_frame is not None:
            # リリース後: 固定位置にオブジェクトを描画
            _draw_box(viewer, self._obj_fixed_frame,
                      size=self._obj_size, rgba=self._obj_rgba)
        else:
            # 初期位置にオブジェクトを描画（まだ掴んでいない）
            _draw_box(viewer, self._obj_initial_frame,
                      size=self._obj_size, rgba=self._obj_rgba)
