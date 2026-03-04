# lstm_encoder_decoder.py
# GPU-safe LSTM encoder/decoder seq2seq with optional early stopping on validation loss.

import copy
import math
import random
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import trange


class lstm_encoder(nn.Module):
    """Encodes time-series sequence."""

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
        # Ensure correct shape
        x = x_input.view(x_input.shape[0], x_input.shape[1], self.input_size)
        lstm_out, hidden = self.lstm(x)
        return lstm_out, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state on the given device."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


class lstm_decoder(nn.Module):
    """Decodes hidden state output by encoder."""

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
        self.linear = nn.Linear(hidden_size, 2)  # (lat, lon)

    def forward(
        self, x_input: torch.Tensor, encoder_hidden_states: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x_input: (batch, input_size)  (here input_size should be 2 for lat/lon)
        encoder_hidden_states: (h_n, c_n)
        returns: (output(batch,2), hidden)
        """
        lstm_out, hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)  # (1, batch, hidden)
        output = self.linear(lstm_out.squeeze(0))  # (batch, 2)
        return output, hidden


class lstm_seq2seq(nn.Module):
    """Train LSTM encoder-decoder and make predictions."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2  # lat/lon

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.decoder = lstm_decoder(input_size=2, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    @staticmethod
    def _decode_rollout_recursive(decoder, decoder_input, decoder_hidden, target_len):
        """
        Recursive rollout decoding (no teacher forcing).
        decoder_input: (batch, 2)
        returns outputs: (target_len, batch, 2)
        """
        batch_size = decoder_input.shape[0]
        device = decoder_input.device
        outputs = torch.zeros(target_len, batch_size, 2, device=device)

        dec_in = decoder_input
        dec_hid = decoder_hidden
        for t in range(target_len):
            dec_out, dec_hid = decoder(dec_in, dec_hid)
            outputs[t] = dec_out
            dec_in = dec_out

        return outputs

    def train_model(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        target_len: int,
        batch_size: int,
        val_input_tensor: Optional[torch.Tensor] = None,
        val_target_tensor: Optional[torch.Tensor] = None,
        max_epochs: int = 500,
        training_prediction: str = "teacher_forcing",   # 'recursive', 'teacher_forcing', 'mixed_teacher_forcing'
        teacher_forcing_ratio: float = 0.5,
        learning_rate: float = 0.01,
        dynamic_tf: bool = False,
        patience: int = 10,            # early stopping patience
        min_delta: float = 0.0,        # minimum improvement
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Trains with optional early stopping (requires val_input_tensor & val_target_tensor).
        Returns history dict with train_losses, val_losses, best_val_loss, best_epoch, epochs_ran.
        """

        device = next(self.parameters()).device

        # Move tensors to model device (THIS FIXES the cpu/cuda mismatch)
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        use_val = (val_input_tensor is not None) and (val_target_tensor is not None)
        if use_val:
            val_input_tensor = val_input_tensor.to(device)
            val_target_tensor = val_target_tensor.to(device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        lat_i, lon_i = 0, 1

        # Use ceil so we don't drop remainder
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

                for b in range(n_batches):
                    start = b * batch_size
                    end = min(start + batch_size, n_train)
                    bs = end - start
                    if bs <= 0:
                        continue

                    input_batch = input_tensor[:, start:end, :]             # (seq_len, bs, features)
                    target_batch = target_tensor[:, start:end, [lat_i, lon_i]]  # (target_len, bs, 2)

                    optimizer.zero_grad()

                    # encode
                    encoder_hidden = self.encoder.init_hidden(bs, device)
                    _, encoder_hidden = self.encoder(input_batch)

                    # decode
                    decoder_input = input_batch[-1, :, [lat_i, lon_i]]  # (bs, 2)
                    decoder_hidden = encoder_hidden

                    if training_prediction == "recursive":
                        outputs = self._decode_rollout_recursive(self.decoder, decoder_input, decoder_hidden, target_len)

                    elif training_prediction == "teacher_forcing":
                        outputs = torch.zeros(target_len, bs, 2, device=device)
                        use_tf = random.random() < teacher_forcing_ratio
                        dec_in = decoder_input
                        dec_hid = decoder_hidden
                        for t in range(target_len):
                            dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                            outputs[t] = dec_out
                            dec_in = target_batch[t] if use_tf else dec_out

                    elif training_prediction == "mixed_teacher_forcing":
                        outputs = torch.zeros(target_len, bs, 2, device=device)
                        dec_in = decoder_input
                        dec_hid = decoder_hidden
                        for t in range(target_len):
                            dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                            outputs[t] = dec_out
                            dec_in = target_batch[t] if (random.random() < teacher_forcing_ratio) else dec_out

                    else:
                        raise ValueError("training_prediction must be 'recursive', 'teacher_forcing', or 'mixed_teacher_forcing'")

                    loss = criterion(outputs, target_batch)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                epoch_loss /= max(n_batches, 1)
                train_losses.append(epoch_loss)

                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - 0.02)

                # Validation + early stopping
                if use_val:
                    self.eval()
                    val_epoch_loss = 0.0

                    with torch.no_grad():
                        for b in range(n_val_batches):
                            start = b * batch_size
                            end = min(start + batch_size, n_val)
                            bs = end - start
                            if bs <= 0:
                                continue

                            v_in = val_input_tensor[:, start:end, :]
                            v_tg = val_target_tensor[:, start:end, [lat_i, lon_i]]

                            v_hidden = self.encoder.init_hidden(bs, device)
                            _, v_hidden = self.encoder(v_in)

                            v_dec_in = v_in[-1, :, [lat_i, lon_i]]
                            v_out = self._decode_rollout_recursive(self.decoder, v_dec_in, v_hidden, target_len)

                            v_loss = criterion(v_out, v_tg)
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
        Single-sample recursive prediction.
        input_tensor: (seq_len, features) tensor
        returns: (target_len, 2) numpy array
        """
        self.eval()
        device = next(self.parameters()).device
        x = input_tensor.to(device).unsqueeze(1)  # (seq_len, 1, features)

        with torch.no_grad():
            _, encoder_hidden = self.encoder(x)

            lat_i, lon_i = 0, 1
            decoder_input = x[-1, :, [lat_i, lon_i]]  # (1, 2)

            outputs = self._decode_rollout_recursive(self.decoder, decoder_input, encoder_hidden, target_len)
            return outputs.squeeze(1).detach().cpu().numpy()