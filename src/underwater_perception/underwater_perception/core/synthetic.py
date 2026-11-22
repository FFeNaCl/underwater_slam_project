"""合成水下场景与合成棋盘格：无相机、无 ROS 的情况下做端到端自测。

用途：
1. **几何检测验证**：渲染红色立方体（真实透视投影 + 可见面明暗 + 内部棱线），
   同时给出精确的轮廓真值，于是可以量化"检出率 / IoU / 顶点数识别"。
2. **标定数学验证**：按针孔 + 畸变模型渲染棋盘格，用真值内参生成视图，
   再跑一遍标定，检查恢复出的 ``fx fy cx cy k1 k2`` 与真值的差距。
3. **跟踪验证**：生成目标逐帧平移的序列，检查 ``track_id`` 保持与误检抑制。

坐标系约定：物体坐标系 ``x`` 向右、``y`` 向下、``z`` 指向观察者背离方向，
与图像坐标一致（便于把投影结果直接当像素用）。这一点与"棋盘格标定里物体坐标
朝向无所谓"的常识不冲突，因为渲染与标定用的是同一套 :func:`chessboard_object_points`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "SyntheticCubeScene",
    "SyntheticBoardScene",
    "render_underwater_background",
    "add_underwater_effects",
    "cube_object_points",
    "render_cube_scene",
    "render_cube_sequence",
    "chessboard_object_points",
    "render_chessboard_scene",
    "make_chessboard_view_set",
    "default_camera_matrix",
]


@dataclass
class SyntheticCubeScene:
    """一帧合成立方体场景。"""

    image: np.ndarray
    #: 轮廓真值（可见面的并集轮廓，顺时针，Nx2 float32）
    silhouette: np.ndarray
    #: 外接框真值 [x1, y1, x2, y2]
    bbox: List[float]
    center: Tuple[float, float]
    rvec: np.ndarray
    tvec: np.ndarray
    #: 可见面索引（0=front,1=back,2=left,3=right,4=top,5=bottom 的约定见 render_cube_scene）
    visible_faces: List[int] = field(default_factory=list)

    @property
    def polygon_area(self) -> float:
        """真值轮廓面积（像素²）。"""
        return float(abs(cv2.contourArea(self.silhouette.astype(np.float32))))


@dataclass
class SyntheticBoardScene:
    """一帧合成棋盘格场景。"""

    image: np.ndarray
    #: 内角点真值，形状 ``(N, 1, 2)`` float32（与 cv2 输出格式一致）
    corners: np.ndarray
    #: 对应的物体点，形状 ``(N, 3)`` float64
    object_points: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    pattern_size: Tuple[int, int]
    square_size: float


def default_camera_matrix(width: int = 1280, height: int = 480, fx: float = 700.0, fy: Optional[float] = None) -> np.ndarray:
    """构造一个合理的内参矩阵（默认主点在图像中心）。"""
    fy = fx if fy is None else fy
    return np.array(
        [[fx, 0.0, width / 2.0 - 0.5], [0.0, fy, height / 2.0 - 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def render_underwater_background(
    width: int,
    height: int,
    seed: int = 0,
    base_color: Sequence[int] = (150, 110, 60),
    texture_strength: float = 0.35,
) -> np.ndarray:
    """生成"水下背景"：蓝绿渐变 + 低频斑驳纹理 + 颗粒噪声 + 暗角。

    Args:
        width: 图像宽。
        height: 图像高。
        seed: 随机种子（保证可复现）。
        base_color: BGR 基色；默认偏蓝绿（水下典型）。
        texture_strength: 纹理强度 0~1。

    Returns:
        ``uint8`` BGR 图像。
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = float(base_color[0])
    base[:, :, 1] = float(base_color[1])
    base[:, :, 2] = float(base_color[2])

    # 上亮下暗的深度渐变（越远越暗、越偏蓝）
    vertical = np.linspace(1.15, 0.65, height, dtype=np.float32).reshape(-1, 1)
    base[:, :, 0] *= vertical * 1.05
    base[:, :, 1] *= vertical
    base[:, :, 2] *= vertical * 0.9

    # 低频斑驳：小尺寸随机场放大，模拟水体不均匀
    coarse_h = max(4, height // 24)
    coarse_w = max(4, width // 24)
    coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype(np.float32)
    texture = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=width / 60.0)
    texture /= (float(np.max(np.abs(texture))) + 1e-6)
    base += (texture[:, :, None] * (60.0 * float(texture_strength)))

    # 高频颗粒噪声
    base += rng.normal(0.0, 3.5, size=base.shape).astype(np.float32)

    # 暗角
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.28 * (radius ** 2), 0.4, 1.0)
    base *= vignette[:, :, None]

    return np.clip(base, 0, 255).astype(np.uint8)


def add_underwater_effects(
    image: np.ndarray,
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    blue_cast: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """叠加水下退化：模糊、噪声、蓝绿偏色（用于鲁棒性测试）。"""
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32)
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(blur_sigma))
    if noise_sigma > 0:
        out += rng.normal(0.0, float(noise_sigma), size=out.shape).astype(np.float32)
    if blue_cast > 0:
        # 蓝绿增益、红通道衰减：模拟水体对长波长的吸收
        out[:, :, 0] *= 1.0 + 0.35 * float(blue_cast)
        out[:, :, 1] *= 1.0 + 0.18 * float(blue_cast)
        out[:, :, 2] *= max(0.15, 1.0 - 0.55 * float(blue_cast))
    return np.clip(out, 0, 255).astype(np.uint8)


def cube_object_points(edge_m: float = 0.05, center: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """返回立方体 8 个顶点在物体坐标系下的坐标（边长 ``edge_m``）。

    顶点顺序（与 :func:`render_cube_scene` 的面索引一致）::

        0: (-,-,-)  1: (+,-,-)  2: (+,+,-)  3: (-,+,-)   前表面 z = -h
        4: (-,-,+)  5: (+,-,+)  6: (+,+,+)  7: (-,+,+)   后表面 z = +h

    ``x`` 向右、``y`` 向下、``z`` 远离观察者。
    """
    h = float(edge_m) / 2.0
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    return np.array(
        [
            [cx - h, cy - h, cz - h],
            [cx + h, cy - h, cz - h],
            [cx + h, cy + h, cz - h],
            [cx - h, cy + h, cz - h],
            [cx - h, cy - h, cz + h],
            [cx + h, cy - h, cz + h],
            [cx + h, cy + h, cz + h],
            [cx - h, cy + h, cz + h],
        ],
        dtype=np.float64,
    )


#: 立方体的 6 个面（顶点索引按逆时针给出，从外部看）
_CUBE_FACES: List[Tuple[int, int, int, int]] = [
    (0, 1, 2, 3),  # front  (z = -h)
    (5, 4, 7, 6),  # back   (z = +h)
    (4, 0, 3, 7),  # left   (x = -h)
    (1, 5, 6, 2),  # right  (x = +h)
    (4, 5, 1, 0),  # top    (y = -h)
    (3, 2, 6, 7),  # bottom (y = +h)
]


def _project(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """投影并返回 ``(N, 2)`` float64 图像坐标。"""
    dist = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)
    return projected.reshape(-1, 2).astype(np.float64)


def render_cube_scene(
    camera_matrix: np.ndarray,
    rvec: Sequence[float],
    tvec: Sequence[float],
    edge_m: float = 0.05,
    image_size: Tuple[int, int] = (1280, 480),
    color_bgr: Sequence[int] = (40, 40, 205),
    dist_coeffs: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    blue_cast: float = 0.0,
    face_shading: Sequence[float] = (1.0, 0.78, 0.62),
    seed: int = 0,
) -> SyntheticCubeScene:
    """渲染一个立方体（可见面不同明暗 + 深色棱线），返回图像与轮廓真值。

    渲染要点：这里**不做**"画个正方形"这种偷懒做法，而是对 8 个顶点做真实
    透视投影、判断哪些面朝向相机、逐面填充不同亮度的颜色、再画棱线。
    因此内部会出现真实的 Y 型顶点与三条棱——这正是几何检测算法要利用的结构，
    用它做验证才有意义。

    Args:
        camera_matrix: 内参。
        rvec, tvec: 物体位姿（Rodrigues 旋转向量 + 平移，单位与 ``edge_m`` 一致）。
        edge_m: 立方体边长（米）。
        image_size: ``(宽, 高)``，与 ``camera_matrix`` 对应。
        color_bgr: 立方体基础颜色（BGR）。
        dist_coeffs: 畸变系数，``None`` 表示无畸变。
        background: 自定义背景；``None`` 时自动生成水下背景。
        blur_sigma / noise_sigma / blue_cast: 水下退化参数。
        face_shading: 三个可见面的亮度系数（按可见顺序依次取用）。
        seed: 随机种子。

    Returns:
        :class:`SyntheticCubeScene`。
    """
    width, height = int(image_size[0]), int(image_size[1])
    if background is None:
        canvas = render_underwater_background(width, height, seed=seed).astype(np.float32)
    else:
        canvas = background.astype(np.float32).copy()
        if canvas.shape[0] != height or canvas.shape[1] != width:
            canvas = cv2.resize(canvas, (width, height)).astype(np.float32)

    rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    points = cube_object_points(edge_m)
    projected = _project(points, rvec_arr, tvec_arr, camera_matrix, dist_coeffs)

    # 判断可见面：投影后多边形方向为逆时针（有向面积为负，y 向下坐标系）则朝向相机
    visible: List[int] = []
    face_polygons: List[Tuple[int, np.ndarray]] = []
    for face_index, indices in enumerate(_CUBE_FACES):
        poly = projected[list(indices)]
        signed_area = 0.0
        for i in range(4):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 4]
            signed_area += x1 * y2 - x2 * y1
        if signed_area < 0.0:
            visible.append(face_index)
            face_polygons.append((face_index, poly))

    base_color = np.array([float(color_bgr[0]), float(color_bgr[1]), float(color_bgr[2])], dtype=np.float32)

    # 用掩码逐面填充（先画所有可见面，再统一做一次模糊，避免边界反复羽化）
    layer = canvas.copy()
    layer_mask = np.zeros((height, width), dtype=np.uint8)
    for order, (_, poly) in enumerate(face_polygons):
        shade = float(face_shading[order % len(face_shading)])
        fill = np.clip(base_color * shade, 0, 255)
        cv2.fillPoly(layer, [poly.astype(np.int32)], color=(float(fill[0]), float(fill[1]), float(fill[2])))
        cv2.fillPoly(layer_mask, [poly.astype(np.int32)], 255)

    canvas = np.where(layer_mask[:, :, None] > 0, layer, canvas)

    # 棱线：画全部 12 条边的投影，内部就会出现 Y 型交点
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    edge_color = tuple(float(max(0.0, c * 0.35)) for c in base_color)
    for a, b in edges:
        p1 = tuple(np.round(projected[a]).astype(int))
        p2 = tuple(np.round(projected[b]).astype(int))
        cv2.line(canvas, p1, p2, edge_color, 2, cv2.LINE_AA)

    image = np.clip(canvas, 0, 255).astype(np.uint8)
    image = add_underwater_effects(image, blur_sigma=blur_sigma, noise_sigma=noise_sigma, blue_cast=blue_cast, seed=seed)

    # 轮廓真值 = 所有可见面顶点的凸包
    if visible:
        hull_input = np.concatenate([projected[list(_CUBE_FACES[i])] for i in visible], axis=0).astype(np.float32)
    else:
        hull_input = projected.astype(np.float32)
    hull = cv2.convexHull(hull_input.reshape(-1, 1, 2)).reshape(-1, 2)
    bbox = [
        float(np.min(hull[:, 0])),
        float(np.min(hull[:, 1])),
        float(np.max(hull[:, 0])),
        float(np.max(hull[:, 1])),
    ]
    center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))

    return SyntheticCubeScene(
        image=image,
        silhouette=hull,
        bbox=bbox,
        center=center,
        rvec=rvec_arr,
        tvec=tvec_arr,
        visible_faces=visible,
    )


def render_cube_sequence(
    camera_matrix: np.ndarray,
    n_frames: int = 12,
    edge_m: float = 0.05,
    image_size: Tuple[int, int] = (1280, 480),
    start: Sequence[float] = (-0.06, 0.0, 0.45),
    step: Sequence[float] = (0.012, 0.0, 0.0),
    rvec: Sequence[float] = (0.35, 0.5, 0.15),
    dist_coeffs: Optional[np.ndarray] = None,
    seed: int = 0,
    **kwargs,
) -> List[SyntheticCubeScene]:
    """生成目标逐帧平移的序列（用于跟踪与序列级测试）。"""
    scenes: List[SyntheticCubeScene] = []
    background = render_underwater_background(int(image_size[0]), int(image_size[1]), seed=seed)
    for index in range(int(n_frames)):
        tvec = [
            float(start[0] + step[0] * index),
            float(start[1] + step[1] * index),
            float(start[2] + step[2] * index),
        ]
        scenes.append(
            render_cube_scene(
                camera_matrix,
                rvec=rvec,
                tvec=tvec,
                edge_m=edge_m,
                image_size=image_size,
                dist_coeffs=dist_coeffs,
                background=background,
                seed=seed + index,
                **kwargs,
            )
        )
    return scenes


def chessboard_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    """棋盘格**内角点**的物体坐标（z=0 平面，x 向右、y 向下）。

    坐标轴约定（**曾经在这里出过 bug，改动前请先读**）：
    ``cols`` 个内角点沿 **x** 方向排列、``rows`` 个沿 **y** 方向排列，
    与 ``cv2.findChessboardCorners(..., (cols, rows))`` 和渲染函数
    :func:`render_chessboard_scene`（沿 x 画 ``cols+1`` 个方格）完全一致。

    开发过程中此处写过"把行索引当 x"，导致物体点阵相对渲染的棋盘整体转置了 90°。
    后果很隐蔽：标定本身仍然能收敛（平面靶标法向不可观测，转置等价于把板子翻面，
    可以被每帧位姿吸收），但所有"检出角点 vs 真值角点"的逐点比较都会差上上百像素，
    极易把排查方向带偏（我当时误判为"检测器锁错了网格"）。因此这里显式写清楚。

    Args:
        cols: 内角点列数（沿 x，``8x6`` 棋盘格传 8）。
        rows: 内角点行数（沿 y，传 6）。
        square_size: 方格边长（米）。

    Returns:
        ``((cols*rows), 3)`` float64 数组，行优先（先沿 x 走完一行再到下一行），
        并以点阵中心为原点。
    """
    points = np.zeros((int(cols) * int(rows), 3), dtype=np.float64)
    grid = np.mgrid[0 : int(rows), 0 : int(cols)].reshape(2, -1).T  # 每行 (iy, ix)
    points[:, 0] = grid[:, 1] * float(square_size)  # x 沿列方向
    points[:, 1] = grid[:, 0] * float(square_size)  # y 沿行方向
    # 以棋盘中心为原点，便于设置位姿
    points[:, 0] -= (int(cols) - 1) * float(square_size) / 2.0
    points[:, 1] -= (int(rows) - 1) * float(square_size) / 2.0
    return points


def render_chessboard_scene(
    camera_matrix: np.ndarray,
    rvec: Sequence[float],
    tvec: Sequence[float],
    cols: int = 8,
    rows: int = 6,
    square_size: float = 0.025,
    image_size: Tuple[int, int] = (1280, 480),
    dist_coeffs: Optional[np.ndarray] = None,
    background: Optional[np.ndarray] = None,
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    quiet_zone_squares: float = 1.2,
    seed: int = 0,
) -> SyntheticBoardScene:
    """渲染棋盘格（逐格填充真实投影四边形），返回图像与内角点真值。

    物件坐标系与 :func:`chessboard_object_points` 一致；棋盘尺寸为
    ``(cols+1) x (rows+1)`` 个方格，即 ``cols x rows`` 个内角点。

    **静区（quiet zone）**：真实打印的标定板四周都留白边，OpenCV 的角点检测器依赖
    这片留白来正确锁定图案。合成图若不留白边，检测器可能锁到偏移/外扩的网格上
    （开发过程中真实遇到过：检测出的 48 个角点整体偏移一格并延伸出板外）。
    因此这里默认画 ``quiet_zone_squares`` 个方格的白色边距。

    Args:
        dist_coeffs: 若给出，则角点位置按畸变模型投影（用于验证畸变系数恢复）。
        quiet_zone_squares: 白边宽度（单位为方格边长）；0 表示不留白边。

    Returns:
        :class:`SyntheticBoardScene`。
    """
    width, height = int(image_size[0]), int(image_size[1])
    if background is None:
        canvas = np.full((height, width, 3), 210, dtype=np.float32)
    else:
        canvas = background.astype(np.float32).copy()
        if canvas.shape[0] != height or canvas.shape[1] != width:
            canvas = cv2.resize(canvas, (width, height)).astype(np.float32)

    rvec_arr = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec_arr = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    squares_x = int(cols) + 1
    squares_y = int(rows) + 1
    board_w = squares_x * float(square_size)
    board_h = squares_y * float(square_size)
    half_w = board_w / 2.0
    half_h = board_h / 2.0

    # 白边：把整块"纸"（棋盘 + 留白）投影后填充白色，再画棋盘
    if quiet_zone_squares > 0:
        margin = float(quiet_zone_squares) * float(square_size)
        paper = np.array(
            [
                [-half_w - margin, -half_h - margin, 0.0],
                [half_w + margin, -half_h - margin, 0.0],
                [half_w + margin, half_h + margin, 0.0],
                [-half_w - margin, half_h + margin, 0.0],
            ],
            dtype=np.float64,
        )
        projected_paper = _project(paper, rvec_arr, tvec_arr, camera_matrix, dist_coeffs)
        cv2.fillPoly(canvas, [projected_paper.astype(np.int32)], color=(248.0, 248.0, 248.0))

    # 每个方格的 4 个角（物体坐标） -> 投影 -> 填充黑白
    for iy in range(squares_y):
        for ix in range(squares_x):
            x0 = -half_w + ix * float(square_size)
            y0 = -half_h + iy * float(square_size)
            corners_obj = np.array(
                [
                    [x0, y0, 0.0],
                    [x0 + float(square_size), y0, 0.0],
                    [x0 + float(square_size), y0 + float(square_size), 0.0],
                    [x0, y0 + float(square_size), 0.0],
                ],
                dtype=np.float64,
            )
            projected = _project(corners_obj, rvec_arr, tvec_arr, camera_matrix, dist_coeffs)
            is_white = (ix + iy) % 2 == 0
            shade = 245.0 if is_white else 25.0
            cv2.fillPoly(canvas, [projected.astype(np.int32)], color=(shade, shade, shade))

    image = np.clip(canvas, 0, 255).astype(np.uint8)
    image = add_underwater_effects(image, blur_sigma=blur_sigma, noise_sigma=noise_sigma, seed=seed)

    object_points = chessboard_object_points(cols, rows, square_size)
    corners = _project(object_points, rvec_arr, tvec_arr, camera_matrix, dist_coeffs).reshape(-1, 1, 2).astype(np.float32)

    return SyntheticBoardScene(
        image=image,
        corners=corners,
        object_points=object_points,
        rvec=rvec_arr,
        tvec=tvec_arr,
        pattern_size=(int(cols), int(rows)),
        square_size=float(square_size),
    )


def make_chessboard_view_set(
    camera_matrix: np.ndarray,
    cols: int = 8,
    rows: int = 6,
    square_size: float = 0.025,
    n_views: int = 15,
    image_size: Tuple[int, int] = (1280, 480),
    dist_coeffs: Optional[np.ndarray] = None,
    distance: float = 0.55,
    blur_sigma: float = 0.6,
    noise_sigma: float = 2.0,
    seed: int = 7,
) -> List[SyntheticBoardScene]:
    """生成一组有姿态多样性的棋盘格视图（供标定自测用）。

    位姿在半球面上随机采样并对准图像中心，符合"手持棋盘在水箱里晃"的实际情况。
    视图数量与角度多样性对可标定性影响极大，``n_views`` 太少时标定结果会漂，
    这也是现场必须采 20~30 个角度的原因。
    """
    rng = np.random.default_rng(seed)
    views: List[SyntheticBoardScene] = []
    background = render_underwater_background(int(image_size[0]), int(image_size[1]), seed=seed)
    for _ in range(int(n_views)):
        # 随机光轴方向上的偏角：限制在 ±45° 内，避免极端掠射角导致角点检出失败
        ax = float(rng.uniform(-0.5, 0.5))
        ay = float(rng.uniform(-0.45, 0.45))
        az = float(rng.uniform(-0.35, 0.35))
        z = float(distance) * (1.0 + float(rng.uniform(-0.12, 0.12)))
        tvec = [ax * z * 0.55, ay * z * 0.35, z]
        rvec = [ax, ay, az]
        views.append(
            render_chessboard_scene(
                camera_matrix,
                rvec=rvec,
                tvec=tvec,
                cols=cols,
                rows=rows,
                square_size=square_size,
                image_size=image_size,
                dist_coeffs=dist_coeffs,
                background=background,
                blur_sigma=blur_sigma,
                noise_sigma=noise_sigma,
                seed=int(rng.integers(0, 10_000)),
            )
        )
    return views
