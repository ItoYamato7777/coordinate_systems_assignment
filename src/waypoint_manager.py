"""
src/waypoint_manager.py
=======================
ウェイポイントの管理クラス。
複数の目標 FRAME を順番に追跡し、手先が閾値以内に入ったら
次のウェイポイントへ切り替える。各ウェイポイントにはオプションで
到達時に実行されるアクション（コールバック）を紐付けできる。
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(_ROOT))
import geo


class WaypointManager:
    """目標 FRAME を順番に管理するクラス。

    Parameters
    ----------
    waypoints : list[tuple[geo.FRAME, Callable | None]]
        目標フレームとオプションのアクションコールバックのリスト
    threshold : float
        位置誤差がこれ以下になったら次のウェイポイントへ進む [m]
    """

    def __init__(
        self,
        waypoints: list[tuple[geo.FRAME, Callable[[], None] | None]],
        threshold: float = 0.01,
    ) -> None:
        if not waypoints:
            raise ValueError("waypoints は 1 つ以上必要です。")
        self._waypoints = waypoints
        self._threshold = threshold
        self._index = 0
        self._action_executed = False  # 現在のウェイポイントのアクションが実行済みか

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    @property
    def current(self) -> geo.FRAME:
        """現在の目標フレームを返す。"""
        return self._waypoints[self._index][0]

    @property
    def index(self) -> int:
        """現在のウェイポイントインデックス。"""
        return self._index

    @property
    def is_finished(self) -> bool:
        """全ウェイポイントを通過したか。"""
        return self._index >= len(self._waypoints)

    def update(self, ee_frame: geo.FRAME) -> bool:
        """手先フレームを受け取り、必要なら次のウェイポイントへ進む。

        Parameters
        ----------
        ee_frame : geo.FRAME
            現在の手先座標系

        Returns
        -------
        bool : ウェイポイントが切り替わった場合 True
        """
        if self.is_finished:
            return False

        pos_ee  = np.array(ee_frame.toarray()[0:3, 3])
        pos_tgt = np.array(self.current.toarray()[0:3, 3])
        err = float(np.linalg.norm(pos_tgt - pos_ee))

        if err < self._threshold:
            # アクションが未実行なら実行
            if not self._action_executed:
                _, action = self._waypoints[self._index]
                if action is not None:
                    action()
                self._action_executed = True

            # 次のウェイポイントへ進む
            if self._index < len(self._waypoints) - 1:
                self._index += 1
                self._action_executed = False
                print(
                    f"[WaypointManager] ウェイポイント切り替え → "
                    f"#{self._index + 1} / {len(self._waypoints)}"
                )
                return True
        return False
