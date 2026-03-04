import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.3)

    def forward(self, x_input):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''

        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


class lstm_decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=0.3)
        self.linear = nn.Linear(hidden_size, 2)    # 2 = lat, lon

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
 
        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class lstm_seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size, hidden_size):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=2, hidden_size=hidden_size)



    def train_model(
        self,
        input_tensor, target_tensor,
        target_len, batch_size,
        val_input_tensor=None, val_target_tensor=None,   
        max_epochs=500,                                   
        training_prediction='recursive',
        teacher_forcing_ratio=0.5,
        learning_rate=0.01,
        dynamic_tf=False,
        patience=10,                                      
        min_delta=0.0,                                    
        save_path=None                                    
    ):
        """
        Early stopping monitors VALIDATION loss.
        Stops after `patience` epochs with no improvement (by at least min_delta).
        """

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        device = next(self.parameters()).device  # cuda:0 if model.to(device) was called

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        if val_input_tensor is not None:
            val_input_tensor = val_input_tensor.to(device)
            val_target_tensor = val_target_tensor.to(device)

        # calculate number of batch iterations
        n_batches = int(input_tensor.shape[1] / batch_size)

        # if validation is provided, compute val batches too
        use_val = (val_input_tensor is not None) and (val_target_tensor is not None)
        if use_val:
            n_val_batches = int(val_input_tensor.shape[1] / batch_size)

        train_losses = []
        val_losses = []

        best_val = float("inf")
        epochs_no_improve = 0
        best_state = None

        lat_i, lon_i = 0, 1

        with trange(max_epochs) as tr:
            for epoch in tr:
                # ---------- TRAIN ----------
                self.train()
                epoch_train_loss = 0.0

                for b in range(n_batches):
                    start = b * batch_size
                    end = start + batch_size

                    input_batch = input_tensor[:, start:end, :]
                    target_batch = target_tensor[:, start:end, [lat_i, lon_i]]

                    outputs = torch.zeros(target_len, batch_size, 2, device=input_batch.device)

                    encoder_hidden = self.encoder.init_hidden(batch_size, device)

                    optimizer.zero_grad()

                    _, encoder_hidden = self.encoder(input_batch)

                    decoder_input = input_batch[-1, :, [lat_i, lon_i]]
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    elif training_prediction == 'teacher_forcing':
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    elif training_prediction == 'mixed_teacher_forcing':
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = target_batch[t, :, :] if random.random() < teacher_forcing_ratio else decoder_output
                    else:
                        raise ValueError("training_prediction must be 'recursive', 'teacher_forcing', or 'mixed_teacher_forcing'")

                    loss = criterion(outputs, target_batch)
                    epoch_train_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                epoch_train_loss /= max(n_batches, 1)
                train_losses.append(epoch_train_loss)

                # optional: dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = max(0.0, teacher_forcing_ratio - 0.02)

                # ---------- VALIDATION ----------
                if use_val:
                    self.eval()
                    epoch_val_loss = 0.0

                    with torch.no_grad():
                        for b in range(n_val_batches):
                            start = b * batch_size
                            end = start + batch_size

                            v_in = val_input_tensor[:, start:end, :]
                            v_tg = val_target_tensor[:, start:end, [lat_i, lon_i]]

                            v_out = torch.zeros(target_len, batch_size, 2, device=v_in.device)

                            v_hidden = self.encoder.init_hidden(batch_size)
                            v_hidden = (v_hidden[0].to(v_in.device),
                                        v_hidden[1].to(v_in.device))

                            _, v_hidden = self.encoder(v_in)

                            # IMPORTANT: validate with NO teacher forcing (recursive)
                            dec_in = v_in[-1, :, [lat_i, lon_i]]
                            dec_hid = v_hidden
                            for t in range(target_len):
                                dec_out, dec_hid = self.decoder(dec_in, dec_hid)
                                v_out[t] = dec_out
                                dec_in = dec_out

                            v_loss = criterion(v_out, v_tg)
                            epoch_val_loss += v_loss.item()

                    epoch_val_loss /= max(n_val_batches, 1)
                    val_losses.append(epoch_val_loss)

                    # ---------- EARLY STOPPING ----------
                    improved = epoch_val_loss < (best_val - min_delta)
                    if improved:
                        best_val = epoch_val_loss
                        epochs_no_improve = 0
                        best_state = copy.deepcopy(self.state_dict())
                        if save_path is not None:
                            torch.save(best_state, save_path)
                    else:
                        epochs_no_improve += 1

                    tr.set_postfix(train=f"{epoch_train_loss:.4f}", val=f"{epoch_val_loss:.4f}", no_improve=epochs_no_improve)

                    if epochs_no_improve >= patience:
                        break
                else:
                    tr.set_postfix(train=f"{epoch_train_loss:.4f}")

        # restore best weights
        if use_val and best_state is not None:
            self.load_state_dict(best_state)

        return {
            "train_losses": np.array(train_losses),
            "val_losses": np.array(val_losses) if use_val else None,
            "best_val_loss": best_val if use_val else None
        }

        '''
        train lstm encoder-decoder
        
        : param input_tensor:              input data with shape (seq_len, # in batch, number features); PyTorch tensor    
        : param target_tensor:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs 
        : param target_len:                number of values to predict 
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''

        
    def predict(self, input_tensor, target_len):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, 2)

        # decode input_tensor
        lat_i, lon_i = 0, 1
        decoder_input = input_tensor[-1, :, [lat_i, lon_i]]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output[0]
            decoder_input = decoder_output

        np_outputs = outputs.detach().numpy()

        return np_outputs
