# batch-norm

## setup

```shell
pip install torch
```

## run

```shell
python main.py
```

## memo

batch norm

- 各層の出力を平均と分散で正規化する。
  - 平均と分散を保持しておき、推論時に使用する。
- 過学習や勾配消失に対して有効
- dropout と同じような効果を持つ。
  - ただ、dropout との併用は注意が必要
