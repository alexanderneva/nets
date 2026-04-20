Practicing neural networks
![forward diff](images/forward_diff_cifar10.png)
![sinusoidal time embeddings](images/embeddings_stacked.png)
![diff pass](images/diff_pass.png)
![reverse diffusion](images/reverse_diff_10.png)
![reverse diffusion 50 epoch](images/reverse_diff_50.png)
```{bash}
[Epoch 50, Batch  200] MSE Loss: 0.0443
[Epoch 50, Batch  400] MSE Loss: 0.0453
[Epoch 50, Batch  600] MSE Loss: 0.0440
```
![reverse diffusion 100 epoch](images/reverse_diff_100.png)
```{bash}
[Epoch 99, Batch  200] MSE Loss: 0.0440
[Epoch 99, Batch  400] MSE Loss: 0.0441
[Epoch 99, Batch  600] MSE Loss: 0.0424
Finished!
[Epoch 100, Batch  200] MSE Loss: 0.0419
[Epoch 100, Batch  400] MSE Loss: 0.0439
[Epoch 100, Batch  600] MSE Loss: 0.0430

```

- loss getting stuck around 0.04
