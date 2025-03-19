#!/usr/bin/env python3
"""
创建MACD策略回测示例

此脚本在examples目录中创建一个简单的MACD策略回测示例，
展示如何使用新的项目结构进行策略回测。

使用方法: python scripts/create_example.py
"""

import logging
import os
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def create_macd_example():
    """创建MACD策略回测示例"""
    # 确保目标目录存在
    example_dir = PROJECT_ROOT / "examples" / "backtest_examples"
    example_dir.mkdir(parents=True, exist_ok=True)

    # 示例文件路径
    example_file = example_dir / "macd_backtest_example.py"

    # 示例代码
    example_code = """#!/usr/bin/env python3
\"\"\"
MACD策略回测示例

此示例展示如何使用StockScan框架创建和回测MACD策略。
\"\"\"

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime

from stockscan.core.data_loader import DataLoader
from stockscan.strategies.technical.macd import MACDStrategy
from stockscan.strategies.backtester import Backtester
from stockscan.utils.visualization import plot_backtest_results

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    \"\"\"解析命令行参数\"\"\"
    parser = argparse.ArgumentParser(description='MACD策略回测示例')
    parser.add_argument('--exchange', type=str, default='NYSE', help='交易所')
    parser.add_argument('--symbol', type=str, default='AAPL', help='股票代码')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='初始资金')
    parser.add_argument('--fast-period', type=int, default=12, help='MACD快线周期')
    parser.add_argument('--slow-period', type=int, default=26, help='MACD慢线周期')
    parser.add_argument('--signal-period', type=int, default=9, help='MACD信号线周期')
    return parser.parse_args()


def run_macd_backtest(args):
    \"\"\"运行MACD策略回测\"\"\"
    logger.info(f"开始MACD策略回测: {args.symbol} 从 {args.start_date}")
    
    # 加载数据
    logger.info("加载市场数据...")
    data_loader = DataLoader(exchange=args.exchange)
    data = data_loader.load_symbol_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if data is None or len(data) == 0:
        logger.error(f"无法加载 {args.symbol} 的数据")
        return
    
    logger.info(f"成功加载数据，共 {len(data)} 条记录")
    
    # 创建策略
    strategy_params = {
        'fast_period': args.fast_period,
        'slow_period': args.slow_period,
        'signal_period': args.signal_period,
    }
    strategy = MACDStrategy(params=strategy_params)
    
    # 创建回测器
    backtester = Backtester(
        strategy=strategy,
        initial_capital=args.initial_capital,
        commission=0.0, # 手续费
        slippage=0.0  # 滑点
    )
    
    # 运行回测
    logger.info("开始回测...")
    results = backtester.run(data)
    
    # 打印回测结果
    logger.info("回测完成，结果如下:")
    logger.info(f"总收益: {results.total_return:.2f}%")
    logger.info(f"年化收益: {results.annualized_return:.2f}%")
    logger.info(f"夏普比率: {results.sharpe_ratio:.2f}")
    logger.info(f"最大回撤: {results.max_drawdown:.2f}%")
    logger.info(f"交易次数: {results.total_trades}")
    logger.info(f"胜率: {results.win_rate:.2f}%")
    
    # 绘制回测结果
    logger.info("绘制回测结果...")
    fig = plot_backtest_results(data, results)
    
    # 保存结果图
    plot_file = f"output/backtest/{args.symbol}_macd_backtest.png"
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig.savefig(plot_file)
    logger.info(f"回测结果图已保存至: {plot_file}")
    
    # 显示图表
    plt.show()
    
    return results


def main():
    \"\"\"主函数\"\"\"
    args = parse_args()
    run_macd_backtest(args)


if __name__ == "__main__":
    main()
"""

    # 创建示例文件
    with open(example_file, "w") as f:
        f.write(example_code)

    # 使文件可执行
    os.chmod(example_file, 0o755)

    logger.info(f"成功创建MACD策略回测示例: {example_file}")
    logger.info("使用方法: python examples/backtest_examples/macd_backtest_example.py")


def create_visualization_template():
    """创建可视化模块模板"""
    # 确保目标目录存在
    utils_dir = PROJECT_ROOT / "stockscan" / "utils"
    utils_dir.mkdir(parents=True, exist_ok=True)

    # 可视化模块路径
    vis_file = utils_dir / "visualization.py"

    # 模块代码
    vis_code = """#!/usr/bin/env python3
\"\"\"
可视化工具模块

此模块提供各种数据可视化功能，包括技术指标、回测结果和绩效指标的可视化。
\"\"\"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import logging

logger = logging.getLogger(__name__)


def plot_price_with_indicators(data, indicators=None, figsize=(12, 8)):
    \"\"\"
    绘制价格图表和技术指标
    
    参数:
        data (pd.DataFrame): 价格数据
        indicators (dict): 指标名称和对应的Series数据
        figsize (tuple): 图表大小
    
    返回:
        fig: matplotlib图表对象
    \"\"\"
    if indicators is None:
        indicators = {}
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
    
    # 主价格图表
    ax1 = plt.subplot(gs[0])
    ax1.plot(data.index, data['Close'], label='Close Price')
    
    # 添加移动平均线等指标
    for name, indicator in indicators.items():
        if name.lower() in ['sma', 'ema', 'wma', 'dema', 'tema', 'trima']:
            ax1.plot(data.index, indicator, label=name)
    
    ax1.set_title('Price Chart with Indicators')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # 成交量图表
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.bar(data.index, data['Volume'])
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    # MACD图表
    if 'macd' in indicators and 'macd_signal' in indicators:
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.plot(data.index, indicators['macd'], label='MACD')
        ax3.plot(data.index, indicators['macd_signal'], label='Signal')
        
        # MACD柱状图
        if 'macd_hist' in indicators:
            ax3.bar(data.index, indicators['macd_hist'], alpha=0.5, label='Histogram')
        
        ax3.set_ylabel('MACD')
        ax3.grid(True)
        ax3.legend(loc='best')
    
    # RSI图表
    if 'rsi' in indicators:
        ax4 = plt.subplot(gs[3], sharex=ax1)
        ax4.plot(data.index, indicators['rsi'], label='RSI')
        ax4.axhline(y=70, color='r', linestyle='--')
        ax4.axhline(y=30, color='g', linestyle='--')
        ax4.set_ylabel('RSI')
        ax4.set_ylim(0, 100)
        ax4.grid(True)
        ax4.legend(loc='best')
    
    # 格式化x轴日期
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_backtest_results(data, results, figsize=(12, 10)):
    \"\"\"
    绘制回测结果
    
    参数:
        data (pd.DataFrame): 价格数据
        results (BacktestResult): 回测结果对象
        figsize (tuple): 图表大小
    
    返回:
        fig: matplotlib图表对象
    \"\"\"
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    # 价格图和交易点
    ax1 = plt.subplot(gs[0])
    ax1.plot(data.index, data['Close'], label='Close Price')
    
    # 标记买入点
    buy_signals = results.signals.get('buy', [])
    if buy_signals:
        buy_dates = [signal['date'] for signal in buy_signals]
        buy_prices = [signal['price'] for signal in buy_signals]
        ax1.scatter(buy_dates, buy_prices, marker='^', color='g', s=100, label='Buy')
    
    # 标记卖出点
    sell_signals = results.signals.get('sell', [])
    if sell_signals:
        sell_dates = [signal['date'] for signal in sell_signals]
        sell_prices = [signal['price'] for signal in sell_signals]
        ax1.scatter(sell_dates, sell_prices, marker='v', color='r', s=100, label='Sell')
    
    ax1.set_title('Backtest Results')
    ax1.set_ylabel('Price')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # 权益曲线
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(results.equity_curve.index, results.equity_curve['equity'], label='Equity')
    
    # 基准线 (Buy & Hold)
    if hasattr(results, 'benchmark'):
        ax2.plot(results.benchmark.index, results.benchmark, label='Benchmark')
    
    ax2.set_ylabel('Equity')
    ax2.grid(True)
    ax2.legend(loc='best')
    
    # 回撤曲线
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.fill_between(
        results.equity_curve.index,
        results.equity_curve['drawdown'].values * 100,
        0,
        color='coral',
        alpha=0.5,
        label='Drawdown'
    )
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_ylim(100, 0)  # 反转y轴，使回撤向下
    ax3.grid(True)
    ax3.legend(loc='best')
    
    # 格式化x轴日期
    plt.gcf().autofmt_xdate()
    
    # 添加性能指标文本框
    textstr = '\\n'.join((
        f'Total Return: {results.total_return:.2f}%',
        f'Annual Return: {results.annualized_return:.2f}%',
        f'Sharpe Ratio: {results.sharpe_ratio:.2f}',
        f'Max Drawdown: {results.max_drawdown:.2f}%',
        f'Win Rate: {results.win_rate:.2f}%',
        f'Profit Factor: {results.profit_factor:.2f}',
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    return fig


def plot_equity_curves(results_dict, title='Equity Curve Comparison', figsize=(12, 6)):
    \"\"\"
    比较多个回测结果的权益曲线
    
    参数:
        results_dict (dict): 策略名称和对应的回测结果对象
        title (str): 图表标题
        figsize (tuple): 图表大小
    
    返回:
        fig: matplotlib图表对象
    \"\"\"
    fig, ax = plt.subplots(figsize=figsize)
    
    for name, results in results_dict.items():
        # 归一化权益曲线，使初始值为100
        equity = results.equity_curve['equity']
        normalized_equity = 100 * equity / equity.iloc[0]
        ax.plot(normalized_equity.index, normalized_equity, label=name)
    
    ax.set_title(title)
    ax.set_ylabel('Equity (normalized to 100)')
    ax.grid(True)
    ax.legend(loc='best')
    
    # 格式化x轴日期
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    return fig


def plot_performance_metrics(results_dict, metrics=None, figsize=(12, 8)):
    \"\"\"
    比较多个回测结果的性能指标
    
    参数:
        results_dict (dict): 策略名称和对应的回测结果对象
        metrics (list): 要比较的指标列表
        figsize (tuple): 图表大小
    
    返回:
        fig: matplotlib图表对象
    \"\"\"
    if metrics is None:
        metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    
    # 提取各策略的指标值
    data = {}
    for metric in metrics:
        data[metric] = []
        for name, results in results_dict.items():
            if hasattr(results, metric):
                value = getattr(results, metric)
                data[metric].append((name, value))
    
    # 创建子图数量
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics):
        names = [item[0] for item in data[metric]]
        values = [item[1] for item in data[metric]]
        
        axes[i].bar(names, values)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].grid(True, axis='y')
        
        # 在条形上方显示数值
        for j, v in enumerate(values):
            axes[i].text(j, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig
"""

    # 创建可视化模块
    with open(vis_file, "w") as f:
        f.write(vis_code)

    logger.info(f"成功创建可视化模块模板: {vis_file}")


def main():
    """主函数"""
    logger.info("开始创建示例文件...")

    # 创建可视化模块模板
    create_visualization_template()

    # 创建MACD策略回测示例
    create_macd_example()

    logger.info("示例文件创建完成!")


if __name__ == "__main__":
    main()
