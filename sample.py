"""
sample.py  ―  同時変換行列 (Homogeneous Transformation Matrix) 学習サンプル
===============================================================================
このスクリプトは geo.FRAME クラスを使って以下の 3 点を学ぶことができます:

  ① 座標系の定義   : FRAME(xyzrpy=[...]) で位置・姿勢を表す
  ② 座標系の連結   : T_A * T_B で「A 座標系上の B」をワールド座標に変換
  ③ アーム動作     : sim.move_arm(F) で手先を目標フレームへ誘導
"""

import math
from pathlib import Path

from geo import FRAME
from src import RobotSimulation

T_work = FRAME(xyzrpy=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0])     # 作業平面
T_offset_1 = FRAME(xyzrpy=[0.0,  0.3, 0.1, 0, 0.3, 0.0])  # 作業平面からのオフセット 1
T_offset_2 = FRAME(xyzrpy=[0.0, -0.1, 0.2, 0, -math.pi/4, 0.5])  # 作業平面からのオフセット 2

# =============================================================================
#  ② 座標系の連結（同時変換行列の掛け算）
#
#  T_target = T_work * T_offset
#  意味: 「T_work 座標系上での T_offset」をワールド座標で表した フレーム
# =============================================================================
T_base2target_1: FRAME = T_work * T_offset_1
T_base2target_2: FRAME = T_work * T_offset_2

print("=== 同時変換行列 学習サンプル ===")
print(f"T_work      xyzrpy = {[round(v, 4) for v in T_work.xyzrpy()]}")
print(f"T_offset_1  xyzrpy = {[round(v, 4) for v in T_offset_1.xyzrpy()]}")
print(f"T_offset_2  xyzrpy = {[round(v, 4) for v in T_offset_2.xyzrpy()]}")
print(f"T_base2target_1  xyzrpy = {[round(v, 4) for v in T_base2target_1.xyzrpy()]}")
print(f"T_base2target_2  xyzrpy = {[round(v, 4) for v in T_base2target_2.xyzrpy()]}")
print()

# =============================================================================
#  ③ move_arm によるシミュレーション実行
#
#  draw_axes : 複数の FRAME を渡すと、それぞれ異なる色で座標軸を描画
#              （原点マーカーの色は内部パレットにより自動割り当て）
#  move_arm  : 呼び出し順に目標ウェイポイントを登録。手先が各目標へ順に移動
#  run()     : MuJoCo Viewer を起動してシミュレーションを開始
# =============================================================================
xml_path = Path(__file__).resolve().parent / "robot_arms" / "universal_robots_ur5e" / "scene.xml"

sim = RobotSimulation(xml_path)

sim.draw_axes(T_base2target_1, T_base2target_2)  # 目標フレームの座標軸を描画（色は自動）
sim.move_arm(T_base2target_1)               # ウェイポイント 1: T_base2target_1 へ移動
sim.move_arm(T_base2target_2)               # ウェイポイント 2: T_base2target_2 へ移動

sim.run()
