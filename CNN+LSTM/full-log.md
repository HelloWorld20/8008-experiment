uv run python CNN+LSTM/learning_pytorch_lstm.py ──(Sat,Mar21)─┘
Using device: mps
Reading files...
Mem. usage decreased to 130.48 Mb (37.5% reduction)
Sell prices has 6841121 rows and 4 columns
Mem. usage decreased to 0.12 Mb (41.9% reduction)
Calendar has 1969 rows and 14 columns
Sales train validation has 30490 rows and 1919 columns
Preprocessing data...
Saved time_series_plot.png

=== Experiment 1: Simple LSTM ===
Saved normalization_dist.png
Data shape: (1884, 28, 1) (1884, 1)
Training Simple LSTM...
0%| | 0/500 [00:00<?, ?it/s]Epoch: 0, loss: 0.38253 valid loss: 0.40843
10%|█████ | 50/500 [00:05<00:50, 8.88it/s]Epoch: 50, loss: 0.16284 valid loss: 0.10278
20%|█████████▊ | 100/500 [00:11<00:45, 8.83it/s]Epoch: 100, loss: 0.16620 valid loss: 0.11022
30%|██████████████▋ | 150/500 [00:17<00:39, 8.91it/s]Epoch: 150, loss: 0.16911 valid loss: 0.09341
40%|███████████████████▌ | 200/500 [00:22<00:33, 8.94it/s]Epoch: 200, loss: 0.15934 valid loss: 0.09701
50%|████████████████████████▌ | 250/500 [00:28<00:28, 8.89it/s]Epoch: 250, loss: 0.16300 valid loss: 0.08482
60%|█████████████████████████████▍ | 300/500 [00:33<00:22, 8.90it/s]Epoch: 300, loss: 0.15817 valid loss: 0.09339
70%|██████████████████████████████████▎ | 350/500 [00:39<00:16, 8.89it/s]Epoch: 350, loss: 0.14729 valid loss: 0.08468
80%|███████████████████████████████████████▏ | 400/500 [00:45<00:11, 8.81it/s]Epoch: 400, loss: 0.14964 valid loss: 0.08695
90%|████████████████████████████████████████████ | 450/500 [00:50<00:05, 8.82it/s]Epoch: 450, loss: 0.15677 valid loss: 0.08651
100%|█████████████████████████████████████████████████| 500/500 [00:56<00:00, 8.84it/s]
Saved prediction_simple_lstm.png

=== Experiment 3: Complex LSTM with Features ===
Feature engineering time: 0.00 min
X_data shape: (1884, 28, 17)
y_data shape: (1884, 1)
Training Complex LSTM...
0%| | 0/500 [00:00<?, ?it/s]Epoch: 0, loss: 0.31525 valid loss: 0.41596
10%|█████ | 50/500 [00:35<05:04, 1.48it/s]Epoch: 50, loss: 0.06944 valid loss: 0.09619
20%|█████████▊ | 100/500 [01:09<04:26, 1.50it/s]Epoch: 100, loss: 0.03701 valid loss: 0.09765
30%|██████████████▋ | 150/500 [01:43<03:55, 1.48it/s]Epoch: 150, loss: 0.00862 valid loss: 0.14616
40%|███████████████████▌ | 200/500 [02:17<03:22, 1.48it/s]Epoch: 200, loss: 0.00433 valid loss: 0.14410
50%|████████████████████████▌ | 250/500 [02:51<02:53, 1.44it/s]Epoch: 250, loss: 0.00416 valid loss: 0.19238
60%|█████████████████████████████▍ | 300/500 [03:25<02:20, 1.43it/s]Epoch: 300, loss: 0.00364 valid loss: 0.17243
70%|██████████████████████████████████▎ | 350/500 [03:59<01:44, 1.43it/s]Epoch: 350, loss: 0.00349 valid loss: 0.18356
80%|███████████████████████████████████████▏ | 400/500 [04:33<01:07, 1.48it/s]Epoch: 400, loss: 0.00359 valid loss: 0.18350
90%|████████████████████████████████████████████ | 450/500 [05:07<00:33, 1.47it/s]Epoch: 450, loss: 0.00333 valid loss: 0.17678
100%|█████████████████████████████████████████████████| 500/500 [05:42<00:00, 1.46it/s]
Loaded best model.
Saved prediction_complex_lstm.png
Done!
