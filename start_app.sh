#!/bin/bash

# スクリプトのあるディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境の有効化
source venv/bin/activate

# アプリケーションの起動
python anylabeling/app.py
