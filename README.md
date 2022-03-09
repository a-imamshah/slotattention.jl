# Slot Attention using Knet Julia

This is an unofficial implementation of "Object-Centric Learning with Slot Attention" using Knet Julia.

Here is the link to the paper: (https://arxiv.org/abs/2006.15055).

## Experimental setup and code
I used the version 1.5.3. of Julia language. All the experiments were conducted using the KUACC cluster. I used
interactive Jupyter Notebooks to implement this project. First, the relevant modules are imported, then layer structures
are defined, and then the model is constructed using these structures. There are few test to check forward pass and the
gradient flow before the training. After this the model is trained and finally it is evaluated and the results are displayed.
The Jupyter Notebooks are given in the GitHub repository. Hyperparameters and training parameters can be adjusted
and running the notebook will start training the model from scratch.
The paper used ARI score to evaluate the performance of the model. I also used the same metrics. The predicted masks
of each slot is binarized and then pixels corresponding to the objects are assigned a label. These masks are compared
with the ground truth masks for evaluation using Julia library function. I contacted one of the authors to understand this
very step. The code is provided in the notebook for evaluation and visual results.


## Computational requirements
The authors of the paper trained the model using 8 Tesla V100 GPUs and it took them around 5 days to train this model.
Whereas, I trained the model using 1 Tesla V100 and it took me around 14 days to train the model for CLEVR dataset.
For the model trained on Multi-dSprites dataset, it took around 9 days to train it. For both models, it takes around 5
minutes to load the model, display the visual results and compute the evaluation metrics.
With my experience on working on object-centric models, I knew that decreasing the instances of the training dataset
has a very negative impact on the training. Therefore, I experimented using full dataset but fewer number of iterations.
Each experiment took little less than 2 days. The reconstruction loss and the model checkpoints were used to monitor
the performance during the training to avoid buggy training.
