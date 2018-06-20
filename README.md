# Project World Models
This repository contains the code base of a project which was executed for the course Natural Computing at the Radboud University. The project was executed by Mick van Hulst, Jorrit van der Laan and Jelle Piepenbrock. The course itself was tutored by Prof. Dr. Marchiori.

The project itself consists of the World Models algorithm which is implemented for the game Space Invaders. The full report is added as a pdf. The original paper can be found via the following [link](https://worldmodels.github.io/). Furthermore, the report which is added describes several code bases which we used to e.g. multithread the training of our controller.

# To run with display on Windows
1. Install vcXsrv (on Windows).
2. Install all dependancies on WSL (bash).
3. pip install gym
4. pip install gym[atari]
5. Run vcXsrv on Windows.
6. run export DISPLAY=:0 (on Bash).
7. Run script

# Run model
0. Run *pip install -r requirements.txt* to install all required packages.
1. Change the config.py file to suit your needs. **For our tutors, we advice not to change the settings and use the number '1' as the variable for folder_save_name**. This results in e.g. *python generate_random.py 1 rand* as a command for step 2.
Note: if you do not have a screen (i.e. training on a server) add '*xvfb-run -a -s "-screen 0 1400x900x24"*' before every Python command. This enables users to train the agent on a computer/server without a monitor.
2. To train the VAE, we need to generate random data. This can be achieved by running: \
*python generate_random.py folder_save_name rand* 
3. Train the VAE: \
*python train_vae.py folder_save_name 1*
The '1' is optional, but allows you to test the VAE after training by decoding latent variables. This enables
users to check what information is encoded by the VAE.
4. The RNN uses latent variables from the VAE, so we need to generate this data from our observations: \
*python generate_random.py folder_save_name rnn*
5. To train the RNN run: \
*python train_rnn.py folder_save_name*
6. To train the controller: \
*timeout x python train_controller.py folder_save_name* \
Note: Due to the usage of multithreading, one has to assign a value to x, which constitutes how long the controller will train in seconds (e.g. *timeout 5 python train_controller.py 1* will result in the controller script running for five seconds). 
7. (Optional) The following command can be used to: 1) visualize the observations which the VAE/RNN encode/predict. 2) Generate new data so that a user can start iterating the entire model. 3) see the best model play. To do this, use the following command: \
*python model.py 1*

If you've executed step 7, then you can start iterating the model by repeating step 3-7.