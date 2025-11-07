## Visualization

This is the visualization of the result of training model

1. Challenge 2: In the first 25 epochs, we combine the metadata to train and after that we just use only the EEG images for training 

- **RMSE between training and testing**. 
![RMSE](rmse_finetune_challenge2.png)

- **Pi Max (The maximum contribution of each mixture)**: This plot help to detect if the MDN suffers from model collapse (rely the prediction on one component only)

    The formula for that is:

    ```math 
    \pi_{\text{max}} = \max_k \pi_k \quad \text{where} \quad \pi_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)}, \; k=1,\dots,K 
    ```

    So it displays the $\max\pi_k$ 
    ![Pi Max](pi_max_finetune_challenge2.png)

    **The Cauchy-Schwarz Inequality**
    ```math
    \left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
    ```


- **Pi entropy:**

    The formula is:

    ```math
    \mathcal{L}_{\text{entropy}} = -\sum_{k=1}^{K} \pi_k \log \pi_k
    \quad \text{where} \quad
    \pi_k = \frac{\exp(z_k)}{\sum_j \exp(z_j)},
    \;\; k=1,\dots,K
    ```

    ![Pi Entropy](pi_entropy_finetune_challenge2.png)

