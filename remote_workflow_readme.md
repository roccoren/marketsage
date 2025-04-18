# 远程工作流配置指南

本文档详细介绍如何配置StockScan系统将工作流结果存储在远程服务器上。系统支持两种主要的远程存储方式：

1. **InfluxDB** - 用于存储时间序列数据、股票数据和模型参数
2. **MLflow** - 用于跟踪实验、存储模型和可视化结果

## 配置方法

有三种方式可以配置远程服务器连接：

### 1. 通过配置文件

编辑 `config/default_config.yaml` 文件中的 `connections` 部分：

```yaml
connections:
  influxdb:
    url: http://your-influxdb-server:8086
    token: your-token
    org: your-org
    bucket: stock_data
  mlflow:
    tracking_uri: http://your-mlflow-server:5000
    experiment_name: marketsage
    artifact_dir: mlruns
```

### 2. 通过环境变量

设置以下环境变量来覆盖配置文件中的设置：

```bash
export INFLUXDB_URL=http://your-influxdb-server:8086
export INFLUXDB_TOKEN=your-token
export INFLUXDB_ORG=your-org
export INFLUXDB_BUCKET=your-bucket
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

### 3. 通过命令行参数

使用我们提供的 `remote_workflow_example.py` 脚本时，可以通过命令行参数指定远程服务器连接：

```bash
python remote_workflow_example.py \
  --storage-backend influxdb \
  --influxdb-url http://your-influxdb-server:8086 \
  --influxdb-token your-token \
  --influxdb-org your-org \
  --influxdb-bucket your-bucket \
  --mlflow-tracking-uri http://your-mlflow-server:5000 \
  --experiment-name your-experiment
```

## 示例用法

### 基本用法

使用配置文件中的默认连接设置：

```bash
python remote_workflow_example.py --storage-backend influxdb
```

### 使用环境变量

```bash
INFLUXDB_URL=http://remote-server:8086 MLFLOW_TRACKING_URI=http://remote-server:5000 \
python remote_workflow_example.py --storage-backend influxdb
```

### 使用命令行参数

```bash
python remote_workflow_example.py \
  --storage-backend influxdb \
  --influxdb-url http://remote-server:8086 \
  --mlflow-tracking-uri http://remote-server:5000
```

## 验证连接

运行脚本后，将输出连接信息，您可以验证是否成功连接到远程服务器：

```
InfluxDB 连接: http://remote-server:8086
MLflow 跟踪服务器: http://remote-server:5000
```

## 工作流结果存储

当工作流运行完成后，结果将存储在以下位置：

1. **InfluxDB存储**:
   - 股票数据存储在指定的bucket中
   - 模型参数和指标存储在同一个bucket的不同measurement中
   - 可以使用InfluxDB UI或API查询这些数据

2. **MLflow存储**:
   - 实验参数和指标记录在MLflow中
   - 可视化图表作为artifacts存储
   - 可以通过MLflow UI查看实验结果

## 故障排除

如果连接远程服务器时遇到问题：

1. 确保远程服务器可访问（检查网络连接、防火墙设置）
2. 验证认证凭据（token、用户名、密码）
3. 检查服务器URL是否正确
4. 查看日志文件获取详细错误信息

如果连接失败，系统将回退到使用本地存储。