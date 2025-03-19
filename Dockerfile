FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements/base.txt /app/requirements/base.txt
RUN pip install --no-cache-dir -r /app/requirements/base.txt

# 复制项目文件
COPY stockscan /app/stockscan
COPY scripts /app/scripts
COPY run_backtest.py /app/run_backtest.py
COPY setup.py pyproject.toml README.md LICENSE /app/

# 创建必要的目录
RUN mkdir -p /app/data/raw /app/data/processed /app/data/features /app/data/labels /app/data/cache \
    /app/logs /app/output /app/config

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 安装包
RUN pip install -e .

# 非root用户运行
RUN useradd -m stockscan
RUN chown -R stockscan:stockscan /app
USER stockscan

# 设置默认命令
ENTRYPOINT ["python", "run_backtest.py"]
CMD ["--help"]