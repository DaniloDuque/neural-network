import torch


class MultilayerPerceptron:
    """
    Perceptron multicapa de 3 capas segun el material del curso (Bishop, 2006).

    Notacion del curso:
      W^o in R^(D+1 x M) — pesos capa entrada -> oculta (fila 0 = bias)
      W^s in R^(M+1 x K) — pesos capa oculta -> salida (fila 0 = bias)

    Parametros
    ----------
    neurons_per_layer : list[int]  [D, M, K]
    alpha             : float      tasa de aprendizaje
    gamma             : float      coeficiente de momentum
    max_weights       : float      rango inicial de pesos uniforme en [-max_weights, max_weights]
    """

    def __init__(self, neurons_per_layer, alpha, gamma, max_weights):
        D, M, K = neurons_per_layer
        self.D = D
        self.M = M
        self.K = K
        self.alpha = alpha
        self.gamma = gamma

        # W^o: (D+1) x M  — fila 0 corresponde al bias W^o_{0,m}
        self.Wo = torch.FloatTensor(D + 1, M).uniform_(-max_weights, max_weights)

        # W^s: (M+1) x K  — fila 0 corresponde al bias W^s_{0,k}
        self.Ws = torch.FloatTensor(M + 1, K).uniform_(-max_weights, max_weights)

        # Terminos de momentum del paso anterior
        self.dWo_prev = torch.zeros_like(self.Wo)
        self.dWs_prev = torch.zeros_like(self.Ws)

        # Salidas de cada capa (se llenan en forward)
        # y^o in R^(n x M),  y^s in R^(n x K)
        self.yo = None
        self.ys = None

        # Deltas de cada capa (se llenan en backpropagate_deltas)
        # delta^o in R^(n x M),  delta^s in R^(n x K)
        self.delta_o = None
        self.delta_s = None

    # ------------------------------------------------------------------
    # Funcion de activacion — sigmoidal para ambas capas (spec del curso)
    # ------------------------------------------------------------------

    @staticmethod
    def _activation(x): return torch.sigmoid(x)

    @staticmethod
    def _activation_derivative(y): return y * (1.0 - y)

    # ------------------------------------------------------------------
    # Forward — Ecuaciones (3), (4), (7), (8) del material del curso
    # ------------------------------------------------------------------

    def forward(self, X):
        """
        Propagacion hacia adelante.

        Capa oculta (ec. 3 y 4):
            p^o_m = sum_{d=0}^{D} W^o_{d,m} * x_d     (x_0 = 1)
            y^o_m = g^o(p^o_m) = sigmoid(p^o_m)

        Capa de salida (ec. 7 y 8):
            p^s_k = sum_{m=0}^{M} W^s_{m,k} * y^o_m   (y^o_0 = 1)
            y^s_k = g^s(p^s_k) = sigmoid(p^s_k)

        Parametros
        ----------
        X : tensor (n, D)

        Retorna
        -------
        ys : tensor (n, K)
        """
        n = X.shape[0]
        ones = torch.ones(n, 1)

        # Entrada aumentada: x_0 = 1 antepuesto  -> shape (n, D+1)
        X_aug = torch.hstack([ones, X])

        # Capa oculta: p^o = X_aug @ W^o,  y^o = sigmoid(p^o)
        po = X_aug @ self.Wo            # (n, M)
        self.yo = self._activation(po)  # (n, M)

        # Entrada aumentada capa oculta: y^o_0 = 1 -> shape (n, M+1)
        yo_aug = torch.hstack([ones, self.yo])

        # Capa de salida: p^s = yo_aug @ W^s,  y^s = sigmoid(p^s)
        ps = yo_aug @ self.Ws           # (n, K)
        self.ys = self._activation(ps)  # (n, K)
        return self.ys

    # ------------------------------------------------------------------
    # Backpropagation — Secciones 1.3.2 y 1.3.3 del material del curso
    # ------------------------------------------------------------------

    def backpropagate_deltas(self, T):
        """
        Calcula los deltas de cada capa segun el material del curso.

        Delta capa de salida (seccion 1.3.2):
            delta^s_k = (y^s_k - t_k) * y^s_k * (1 - y^s_k)

        Delta capa oculta (seccion 1.3.3):
            delta^o_m = ( sum_{k} delta^s_k * W^s_{m,k} ) * y^o_m * (1 - y^o_m)

        Nota: se excluye la fila 0 de W^s (bias) al propagar hacia atras,
        pues y^o_0 = 1 es constante y no tiene delta asociado.

        Parametros
        ----------
        T : tensor (n, K)  etiquetas objetivo
        """
        # delta^s: (n, K)
        self.delta_s = (self.ys - T) * self._activation_derivative(self.ys)

        # delta^o: (n, M)
        # Ws[1:] excluye la fila del bias -> shape (M, K)
        self.delta_o = (self.delta_s @ self.Ws[1:].T) * self._activation_derivative(self.yo)

    # ------------------------------------------------------------------
    # Actualizacion de pesos — Ecuaciones (13) y (22) del material
    # FIX: se normaliza por n para que alpha sea independiente del
    #      tamanio del batch y el entrenamiento sea estable.
    # ------------------------------------------------------------------

    def update_weights(self, X):
        n = X.shape[0]
        ones = torch.ones(n, 1)

        X_aug  = torch.hstack([ones, X])           # (n, D+1)
        yo_aug = torch.hstack([ones, self.yo])     # (n, M+1)

        # Gradientes promediados sobre todas las muestras  ← FIX
        grad_Ws = (yo_aug.T @ self.delta_s) / n    # (M+1, K)
        grad_Wo = (X_aug.T  @ self.delta_o) / n    # (D+1, M)

        # Actualizacion con momentum
        dWs = self.alpha * grad_Ws + self.gamma * self.dWs_prev
        dWo = self.alpha * grad_Wo + self.gamma * self.dWo_prev

        # Descenso: W = W - dW  (signo negativo segun ec. 13 del curso)
        self.Ws -= dWs
        self.Wo -= dWo

        self.dWs_prev = dWs
        self.dWo_prev = dWo

    # ------------------------------------------------------------------
    # Error MSE — E(w) = (1/2) * sum ||y - t||^2  (seccion 1.2)
    # ------------------------------------------------------------------

    def get_error(self, X, T):
        predictions = self.forward(X)
        return torch.mean((T - predictions) ** 2).item()

    # ------------------------------------------------------------------
    # Entrenamiento — Seccion 1.3 y algoritmo de la seccion 1.4
    # ------------------------------------------------------------------

    def train_mlp(self, num_epochs, X, T, X_val=None, T_val=None):
        train_errors = []
        val_errors   = []

        for _ in range(num_epochs):
            self.forward(X)
            self.backpropagate_deltas(T)
            self.update_weights(X)

            train_errors.append(self.get_error(X, T))

            if X_val is not None and T_val is not None:
                val_errors.append(self.get_error(X_val, T_val))

        return train_errors, val_errors

    def predict(self, X, threshold=0.5): return (self.forward(X) >= threshold).int()
