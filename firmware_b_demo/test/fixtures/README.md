# 固定数值对照材料

这些文件是已部署模型的复现依赖，不是新增采集或独立验收集。

- `recognition/time`、`recognition/spectrum`：来自 `20261017_three_point_wood_id` 导出目录的固定三点输入、预处理和预期分数，每种模型六组。
- `quality`：来自 `20261018_normal_only_quality_v1` 的质量模型及训练、校准、正常测试、异常测试四个互不重叠的记录文件，共300条。
- `sha256.json`：以上文件的 SHA-256，用于确认复制完整性。

训练与校准记录用于重建质量模型的 Python 对照，不能计为留出测试。五条指定异常未参与质量参考拟合或阈值校准；其余记录暂按正常。分类模型的选择和场地容错档已经参考既有测试结果，限制见固件说明。

`verify_models.py`、`verify_quality.py` 与 `venue_sensitivity.py` 默认直接读取这里，不依赖被忽略的 `cnn_analysis/runs` 目录。
