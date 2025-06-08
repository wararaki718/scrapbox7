import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, model: nn.Module, dataloader: DataLoader, n_epochs: int=100) -> nn.Module:
        # オプティマイザ
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 損失関数
        # NLLLossは、LogSoftmaxの出力に対して使用
        criterion = nn.NLLLoss()

        # デバイスの設定
        model.to(self._device)

        # 学習ループ
        for epoch in range(1, n_epochs+1):
            total_loss = 0.0
            for context_words_batch, target_words_batch in dataloader:
                context_words_batch: torch.Tensor = context_words_batch.to(self._device)
                target_words_batch: torch.Tensor = target_words_batch.to(self._device)

                optimizer.zero_grad()

                # モデルのフォワードパス
                log_probs = model(context_words_batch)

                # 損失の計算
                loss: torch.Tensor = criterion(log_probs, target_words_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}", flush=True)

        return model
