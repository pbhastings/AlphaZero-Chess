# **Applying the AlphaZero algorithm to chess**

This project implements [DeepMind's AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) to train a chess-playing neural network via reinforcement learning. It trains a simpler version of the model from the paper (fewer layers and shallower MCTS depth) to save on training and prediction time.
In the future, I plan on first training the policy and value heads of the model on grandmaster games before the reinforcement learning stage to see if it improves results. (This kind of contradicts AlphaZero's "tabula rasa" philosophy, but training from scratch takes a long time...)

Train the a model using the main.ipynb file, and run the trained model on a given position in the engine.ipynb file.
The project uses keras and the python chess library.

# Sources:

[AlphaZero Paper Summary](https://arxiv.org/pdf/1712.01815.pdf)

[AlphaGo Zero Paper](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)

[Monte Carlo Tree Search Explanation](https://www.youtube.com/watch?v=UXW2yZndl7U)