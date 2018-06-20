# Project World Models
This repository contains the code base of a project which was executed for the course Natural Computing at the Radboud University. The project was executed by Mick van Hulst, Jorrit van der Laan and Jelle Piepenbrock. The course itself was tutored by Prof. Dr. Marchiori.

The project itself consists of the World Models algorithm which is implemented for the game Space Invaders. The full report is added as a pdf.

# To run windows with screen
1. Install vcXsrv (on Windows).
2. Install all dependancies on WSL (bash).
3. pip install gym
4. pip install gym[atari]
5. Run vcXsrv on Windows.
6. run export DISPLAY=:0 (on Bash).
7. Run script

# Run model
0. Run *pip install -r requirements.txt* to install all required packages.
1. Change the config.py file to suit your needs.
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
*python train_controller.py folder_save_name*
7. (Optional) If you want to visualize VAE and/or RNN images, and/or want to generate new data so that you can iterate, run the following: \
*python model.py 1*

If you've executed step 7, then you can iterate the environment by starting at step 3