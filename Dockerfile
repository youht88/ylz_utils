FROM langchain/langgraph-api:3.11

RUN apt-get update
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config repositories.tuna https://pypi.tuna.tsinghua.edu.cn/simple
ADD ./pyproject.toml /deps/ylz_utils/
# 禁止poetry自动创建虚拟环境
RUN poetry config virtualenvs.create false
# 安装依赖包
RUN poetry install -C /deps/ylz_utils --no-root
ADD ./ylz_utils/config.yaml /root/.ylz_utils/

ADD . /deps/ylz_utils

#RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -c /api/constraints.txt -e /deps/*

ENV LANGSERVE_GRAPHS='{"life": "/deps/ylz_utils/graph_cloud.py:get_life_graph", "test": "/deps/ylz_utils/graph_cloud.py:get_test_graph"}'

WORKDIR /deps/ylz_utils
