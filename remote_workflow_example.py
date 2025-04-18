#!/usr/bin/env python3
"""
示例脚本：演示如何将工作流程结果存储在远程服务器上。
这个示例展示了如何配置 InfluxDB 和 MLflow 来存储工作流程结果。
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入模块
from modules.config_manager import load_config_file, get_config
from modules.logging_manager import get_logger
from modules.storage import StorageManager
from modules.data_acquisition import DataAcquisitionManager
from modules.forecasting import ForecastingManager
from modules.backtesting import BacktestingManager
from modules.workflow_manager import WorkflowManager, DataAcquisitionStep, ForecastingStep, BacktestingStep, VisualizationStep

# 配置日志
logger = get_logger(__name__)


def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="远程工作流示例")
    
    # 配置参数
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="配置文件路径")
    
    # 数据获取参数
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOG",
                        help="股票代码（逗号分隔）")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                        help="历史数据开始日期 (YYYY-MM-DD)")
    
    # 存储参数
    parser.add_argument("--storage-backend", type=str, default="influxdb",
                        help="存储后端 (csv, sqlite, influxdb)")
    
    # 远程服务器参数
    parser.add_argument("--influxdb-url", type=str, default=None,
                        help="InfluxDB 服务器 URL")
    parser.add_argument("--influxdb-token", type=str, default=None,
                        help="InfluxDB 认证令牌")
    parser.add_argument("--influxdb-org", type=str, default=None,
                        help="InfluxDB 组织")
    parser.add_argument("--influxdb-bucket", type=str, default=None,
                        help="InfluxDB 存储桶")
    
    # MLflow 参数
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None,
                        help="MLflow 跟踪服务器 URI")
    parser.add_argument("--experiment-name", type=str, default="remote_workflow_test",
                        help="MLflow 实验名称")
    
    return parser.parse_args()


def set_environment_variables(args):
    """根据命令行参数设置环境变量。"""
    # 设置 InfluxDB 环境变量
    if args.influxdb_url:
        os.environ["INFLUXDB_URL"] = args.influxdb_url
    if args.influxdb_token:
        os.environ["INFLUXDB_TOKEN"] = args.influxdb_token
    if args.influxdb_org:
        os.environ["INFLUXDB_ORG"] = args.influxdb_org
    if args.influxdb_bucket:
        os.environ["INFLUXDB_BUCKET"] = args.influxdb_bucket
    
    # 设置 MLflow 环境变量
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri


def main():
    """主执行函数。"""
    args = parse_arguments()
    
    # 设置环境变量
    set_environment_variables(args)
    
    # 加载配置
    load_config_file(args.config)
    logger.info(f"从 {args.config} 加载配置")
    
    # 获取股票代码
    symbols = [symbol.strip() for symbol in args.symbols.split(",")]
    logger.info(f"处理 {len(symbols)} 个股票: {', '.join(symbols)}")
    
    # 初始化管理器
    storage_manager = StorageManager(backend_type=args.storage_backend)
    data_manager = DataAcquisitionManager(storage_manager=storage_manager)
    forecasting_manager = ForecastingManager()
    backtesting_manager = BacktestingManager()
    
    # 打印远程服务器连接信息
    from modules.connection_config import get_influxdb_config, get_mlflow_config
    influxdb_config = get_influxdb_config()
    mlflow_config = get_mlflow_config()
    
    logger.info(f"InfluxDB 连接: {influxdb_config['url']}")
    logger.info(f"MLflow 跟踪服务器: {mlflow_config['tracking_uri']}")
    
    # 创建工作流
    workflow = WorkflowManager(
        experiment_name=args.experiment_name,
        tracking_uri=mlflow_config['tracking_uri'],
        storage_backend=args.storage_backend
    )
    
    # 添加步骤
    workflow.add_step(
        DataAcquisitionStep(data_manager, symbols, args.start_date, store=True)
    )
    
    workflow.add_step(
        ForecastingStep(forecasting_manager, model_type="ensemble", optimize=False)
    )
    
    workflow.add_step(
        BacktestingStep(backtesting_manager, strategy="momentum", optimize=False)
    )
    
    workflow.add_step(
        VisualizationStep(output_dir="visualizations")
    )
    
    try:
        logger.info("运行工作流...")
        result = workflow.run()
        logger.info("工作流成功完成")
        
        # 打印摘要
        print("\n工作流摘要:")
        print(f"- 处理的股票: {len(symbols)}")
        print(f"- 存储后端: {args.storage_backend}")
        
        if "data" in result:
            print(f"- 获取了 {len(result['data'])} 个股票的数据")
        
        if "forecasts" in result:
            print(f"- 为 {len(result['forecasts'])} 个股票生成了预测")
            # 打印预测变化
            for symbol, forecast in result['forecasts'].items():
                print(f"  - {symbol}: {forecast['forecast_change_percent']:.2f}% 在 {forecast['forecast_horizon']} 天内的变化")
        
        if "backtest_results" in result:
            print(f"- 完成了 {len(result['backtest_results'])} 个股票的回测")
            # 打印回测收益
            for symbol, metrics in result['backtest_results'].items():
                print(f"  - {symbol}: {metrics['total_return']*100:.2f}% 收益率, 夏普比率: {metrics['sharpe_ratio']:.2f}")
        
        if "visualization_paths" in result:
            print(f"- 创建了 {len(result['visualization_paths'])} 个可视化")
            for name, path in result["visualization_paths"].items():
                print(f"  - {name}: {path}")
        
        print("\n在 MLflow UI 中查看详细结果:")
        print(f"- MLflow UI: {mlflow_config['tracking_uri']}")
        print("- 实验名称: " + args.experiment_name)
    
    except Exception as e:
        logger.error(f"运行工作流时出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()