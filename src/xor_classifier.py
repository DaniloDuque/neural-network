## @file xor_classifier.py
#  @brief Entrenamiento y evaluación del clasificador XOR usando MultilayerPerceptron.

import torch
from multilayer_perceptron import MultilayerPerceptron

CONFIGS = [
    (0.01, 0.0),
    (0.10, 0.0),
    (0.50, 0.0),
    (0.01, 0.9),
    (0.10, 0.9),
    (0.50, 0.9),
]

CONVERGENCE_THRESHOLD = 1e-3


def train_xor_configuration(alpha, gamma, num_epochs=50000, hidden_neurons=2, max_weights=0.1):
    """
    @brief Entrena una red MLP sobre el dataset XOR con los hiperparámetros dados.

    @param alpha          Tasa de aprendizaje.
    @param gamma          Coeficiente de momentum.
    @param num_epochs     Número de épocas de entrenamiento.
    @param hidden_neurons Número de neuronas en la capa oculta (M).
    @param max_weights    Rango inicial de pesos en [-max_weights, max_weights].

    @return dict con claves: mlp, errors, final_error, alpha, gamma.
    """
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    T = torch.tensor([[0.], [1.], [1.], [0.]])

    mlp = MultilayerPerceptron([2, hidden_neurons, 1], alpha=alpha, gamma=gamma,
                               max_weights=max_weights)
    errors, _ = mlp.train_mlp(num_epochs, X, T)

    return {
        "mlp": mlp,
        "errors": errors,
        "final_error": errors[-1],
        "alpha": alpha,
        "gamma": gamma,
    }


def convergence_epoch(errors, threshold=CONVERGENCE_THRESHOLD):
    """
    @brief Retorna la primera época donde el error cae por debajo del umbral.

    @param errors    Lista de errores por época.
    @param threshold Umbral de convergencia.

    @return Índice de época (int) o None si nunca converge.
    """
    for epoch, e in enumerate(errors):
        if e < threshold:
            return epoch
    return None
