# YOLO 模型权重目录

## 当前状态（2026-11-19）

| 文件 | 说明 |
|---|---|
| `best.pt` | 单类 `rubbish`，YOLOv8n，来源 `runs/yolo/rubbish_v2/weights/last.pt` |
| `best.onnx` | 由上述权重导出（输入 1x3x640x640，输出 1x5x8400） |
| `backup/best_vrx_original.onnx` | 原 VRX/水面 6 类权重（10 维输出），已保留 |
| `best_metadata.json` | 类别、训练参数、ONNX 输入输出形状 |

> **重要**：训练集只有 10 张、验证集仅 3 张。`best.pt` 是按验证集选的，
> 在这么小的验证集上会选到退化模型（置信度≈0.01）；最终使用的是 `last.pt`。
> 本机自测：train 10/10、valid 3/3 命中（conf≥0.25），但这只是**同一段视频的
> 近邻帧**，不能代表真实泛化精度。正式实验请补采至少每类 200 个实例后重训。

`config/yolo_real_camera.yaml` 已指向：

```yaml
model_path: /home/hu/underwater_slam_project/models/best.pt
```

## 训练完成后

```bash
cp /home/hu/underwater_slam_project/runs/yolo/underwater_real_v1/weights/best.pt \
   /home/hu/underwater_slam_project/models/best.pt

# 可选：导出 ONNX 作为 CPU 部署备选
yolo export \
  model=/home/hu/underwater_slam_project/models/best.pt \
  format=onnx imgsz=640
```

然后修改 `config/yolo_real_camera.yaml`：

```yaml
model_path: /home/hu/underwater_slam_project/models/best.pt
```

完整采集/标注/训练流程见 `src/underwater_perception/docs/YOLO_REAL_DATA.md`。
