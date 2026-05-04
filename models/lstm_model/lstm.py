import copy
import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange


class lstm_encoder(nn.Module):
    """Encodes an input time-series sequence."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x_input: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x_input: (seq_len, batch, input_size)
        returns: (lstm_out, (h_n, c_n))
        """
        if x_input.dim() != 3:
            raise ValueError(f"x_input must be 3D, got shape {tuple(x_input.shape)}")
        if x_input.shape[2] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got last dim={x_input.shape[2]}"
            )
        return self.lstm(x_input)

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0


class lstm_decoder(nn.Module):
    """
    Decoder
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(
        self, x_input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x_input: (batch, input_size)
        hidden: (h_n, c_n)
        returns: (output(batch, output_size), hidden)
        """
        if x_input.dim() != 2:
            raise ValueError(f"x_input must be 2D, got shape {tuple(x_input.shape)}")
        lstm_out, hidden = self.lstm(x_input.unsqueeze(0), hidden)  # (1, batch, hidden)
        output = self.linear(lstm_out.squeeze(0))                   # (batch, output_size)
        return output, hidden


class lstm_seq2seq(nn.Module):
    """
    Seq2seq LSTM that predicts the target tensor directly.
    For your project, target_tensor should already be [dlat, dlon].
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        decoder_feature_indices: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if decoder_feature_indices is None:
            decoder_feature_indices = tuple(range(input_size))
        self.decoder_feature_indices = tuple(decoder_feature_indices)
        self.decoder_input_size = len(self.decoder_feature_indices)

        self.encoder = lstm_encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = lstm_decoder(
            input_size=self.decoder_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.target_to_decoder = nn.Linear(output_size, self.decoder_input_size)

    def _make_initial_decoder_input(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Use the last observed feature vector subset as the first decoder input."""
        return input_batch[-1, :, list(self.decoder_feature_indices)]

    def _make_next_decoder_input(self, pred_out: torch.Tensor, prev_decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Project the predicted 2-d target back into decoder-input space.
        """
        projected = self.target_to_decoder(pred_out)
        return 0.5 * projected + 0.5 * prev_decoder_input

    def _decode_rollout_recursive(
        self,
        decoder_input: torch.Tensor,
        decoder_hidden: Tuple[torch.Tensor, torch.Tensor],
        target_len: int,
    ) -> torch.Tensor:
        batch_size = decoder_input.shape[0]
        device = decoder_input.device
        outputs = torch.zeros(target_len, batch_size, self.output_size, device=device)

        dec_in = decoder_input
        dec_hid = decoder_hidden
        for t in range(target_len):
            dec_out, dec_hid = self.decoder(dec_in, dec_hid)
            outputs[t] = dec_out
            dec_in = self._make_next_decoder_input(dec_out, dec_in)

        return outputs

    def _run_decoder(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        target_len: int,
        training_prediction: str,
        teacher_forcing_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = input_batch.shape[1]
        device = input_batch.device

        _, encoder_hidden = self.encoder(input_batch)
        decoder_input = self._make_initial_decoder_input(input_batch)
        decoder_hidden = encoder_hidden

        if training_prediction == "recursive":
            outputs = self._decode_rollout_recursive(decoder_input, decoder_hidden, target_len)

        elif training_prediction == "teacher_forcing":
            outputs = torch.zeros(target_len, bs, self.output_size, device=device)
            use_tf = random.random() < teacher_forcing_ratio
            dec_in = decoder_input
            dec_hid = decoder_hidden
            for t in range(target_len):
                dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                outputs[t] = dec_out
                if use_tf:
                    dec_in = self._make_next_decoder_input(target_batch[t], dec_in)
                else:
                    dec_in = self._make_next_decoder_input(dec_out, dec_in)

        elif training_prediction == "mixed_teacher_forcing":
            outputs = torch.zeros(target_len, bs, self.output_size, device=device)
            dec_in = decoder_input
            dec_hid = decoder_hidden
            for t in range(target_len):
                dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                outputs[t] = dec_out
                if random.random() < teacher_forcing_ratio:
                    dec_in = self._make_next_decoder_input(target_batch[t], dec_in)
                else:
                    dec_in = self._make_next_decoder_input(dec_out, dec_in)

        else:
            raise ValueError("training_prediction must be 'recursive', 'teacher_forcing', or 'mixed_teacher_forcing'")

        return outputs, target_batch

    def train_model(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        target_len: int,
        batch_size: int,
        val_input_tensor: Optional[torch.Tensor] = None,
        val_target_tensor: Optional[torch.Tensor] = None,
        max_epochs: int = 500,
        training_prediction: str = "teacher_forcing",
        teacher_forcing_ratio: float = 0.5,
        learning_rate: float = 0.001,
        dynamic_tf: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        save_path: Optional[str] = None,
        grad_clip: Optional[float] = 1.0,
        shuffle_batches: bool = True,
    ) -> Dict[str, Any]:
        """
        input_tensor:  (seq_len, N, input_size)
        target_tensor: (target_len, N, output_size)
        """
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        use_val = (val_input_tensor is not None) and (val_target_tensor is not None)
        if use_val:
            val_input_tensor = val_input_tensor.to(device)
            val_target_tensor = val_target_tensor.to(device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        n_train = input_tensor.shape[1]
        n_batches = math.ceil(n_train / batch_size)

        if use_val:
            n_val = val_input_tensor.shape[1]
            n_val_batches = math.ceil(n_val / batch_size)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None
        best_epoch = -1
        epochs_no_improve = 0

        with trange(max_epochs) as tr:
            for epoch in tr:
                self.train()
                epoch_loss = 0.0

                indices = torch.randperm(n_train, device=device) if shuffle_batches else torch.arange(n_train, device=device)

                for b in range(n_batches):
                    batch_idx = indices[b * batch_size:(b + 1) * batch_size]
                    if batch_idx.numel() == 0:
                        continue

                    input_batch = input_tensor[:, batch_idx, :]
                    target_batch = target_tensor[:, batch_idx, :]

                    optimizer.zero_grad()
                    outputs, target_batch_used = self._run_decoder(
                        input_batch=input_batch,
                        target_batch=target_batch,
                        target_len=target_len,
                        training_prediction=training_prediction,
                        teacher_forcing_ratio=teacher_forcing_ratio,
                    )

                    loss = criterion(outputs, target_batch_used)
                    loss.backward()
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()

                epoch_loss /= max(n_batches, 1)
                train_losses.append(epoch_loss)

                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - 0.02)

                if use_val:
                    self.eval()
                    val_epoch_loss = 0.0
                    with torch.no_grad():
                        for b in range(n_val_batches):
                            start = b * batch_size
                            end = min(start + batch_size, n_val)
                            if end <= start:
                                continue

                            v_in = val_input_tensor[:, start:end, :]
                            v_tg = val_target_tensor[:, start:end, :]

                            v_out, v_tg_used = self._run_decoder(
                                input_batch=v_in,
                                target_batch=v_tg,
                                target_len=target_len,
                                training_prediction="recursive",
                                teacher_forcing_ratio=0.0,
                            )
                            v_loss = criterion(v_out, v_tg_used)
                            val_epoch_loss += v_loss.item()

                    val_epoch_loss /= max(n_val_batches, 1)
                    val_losses.append(val_epoch_loss)

                    improved = val_epoch_loss < (best_val_loss - min_delta)
                    if improved:
                        best_val_loss = val_epoch_loss
                        best_epoch = epoch
                        best_state = copy.deepcopy(self.state_dict())
                        epochs_no_improve = 0
                        if save_path:
                            torch.save(best_state, save_path)
                    else:
                        epochs_no_improve += 1

                    tr.set_postfix(train=f"{epoch_loss:.4f}", val=f"{val_epoch_loss:.4f}", no_improve=epochs_no_improve)
                    if epochs_no_improve >= patience:
                        break
                else:
                    tr.set_postfix(train=f"{epoch_loss:.4f}")

        if use_val and best_state is not None:
            self.load_state_dict(best_state)

        return {
            "train_losses": np.array(train_losses),
            "val_losses": np.array(val_losses) if use_val else None,
            "best_val_loss": best_val_loss if use_val else None,
            "best_epoch": best_epoch if use_val else None,
            "epochs_ran": len(train_losses),
        }

    def predict(self, input_tensor: torch.Tensor, target_len: int) -> np.ndarray:
        """
        Returns direct model outputs in target space.
        For your project, that means predicted [dlat, dlon].
        """
        self.eval()
        device = next(self.parameters()).device

        x = input_tensor.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"input_tensor must be 2D or 3D, got shape {tuple(x.shape)}")

        with torch.no_grad():
            _, encoder_hidden = self.encoder(x)
            decoder_input = self._make_initial_decoder_input(x)
            outputs = self._decode_rollout_recursive(decoder_input, encoder_hidden, target_len)
            return outputs.squeeze(1).detach().cpu().numpy()