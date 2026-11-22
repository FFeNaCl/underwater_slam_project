# 训练结果（节选）

`runs/yolo/rubbish_v2/` 保留了训练曲线、混淆矩阵和验证集可视化，用于证明流程可运行。
训练权重已统一放到 `models/best.pt` / `models/best.onnx`，这里没有重复放权重。

注意：数据集只有 13 张图（10 train / 3 valid），指标没有统计意义；
正式实验请补采真实水下图片后重训。
