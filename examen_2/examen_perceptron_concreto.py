import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    return StandardScaler, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Examen 2: perceptrón de una capa, regresión lineal y regresión logística

    **Materia:** Matemáticas Aplicadas a Ciencia de Datos

    Completa las celdas marcadas con `TODO` y escribe tus respuestas en las celdas Markdown debajo de cada sección. Puedes consultar `perceptron_wine_referencia.ipynb` de la tarea de repaso. Las funciones de python que hicimos durante clase para el entrenamiento del perceptron se encuentran al final de este notebook de marimo.

    Usaremos el dataset **Concrete Compressive Strength**, que contiene 1,030 observaciones y ocho variables de entrada: siete cantidades de ingredientes y la edad del concreto en días. La variable objetivo es la resistencia a compresión en MPa.

    En este examen puedes entrenar y evaluar con los mismos datos. El objetivo es comparar la estructura matemática y el entrenamiento de los modelos.

    [Fuente: UCI Machine Learning Repository, CC BY 4.0](https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Carga, variables objetivo y estandarización

    Se usa `target` como la variable objetivo numérica. Se crea `target_cat` como la variable objetivo categórica, con valor 1 si es mayor que su mediana y 0 en los demás casos. Se estandarizan los datos con un escalador `StandardScaler`

    Se separan las características `X`, target numérico `y_num` y target categórico `y_cat`. Se estandariza únicamente `X` con `StandardScaler` en una nueva variable `X_std`.
    """)
    return


@app.cell
def _(StandardScaler, np, pd):
    url = "https://raw.githubusercontent.com/IvTole/Matematicas_Aplicadas_Ciencia_De_Datos_CUGDL/refs/heads/main/data/concrete/concrete.csv"

    # renombrar target
    df = pd.read_csv(url).rename(columns={"Concrete compressive strength": "target"})

    # crear nuevo target (binario)
    target_threshold = df["target"].median()
    df["target_cat"] = np.where(df["target"] > target_threshold, 1, 0)

    # Variables a utilizar
    X= df.drop(columns=["target", "target_cat"])
    y_num = df["target"]
    y_cat = df["target_cat"]

    # Proceso de estandarización (únicamente sobre X)
    # Se crean datos de entrada estandarizados X_std
    scaler = StandardScaler()
    X_std = pd.DataFrame(data=scaler.fit_transform(X),
                         columns=X.columns
                        )
    X_std.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Dimensiones e inicialización

    Organiza `X_array` con una observación por columna y ambos targets con una sola fila. Usa la función `layer_sizes` e `initialize_parameters`.

    **Pregunta 1.** ¿Cuáles son las dimensiones de `X_array`, `W`, `b` y `Y_hat` para el conjunto completo? Explica cómo se puede sumar `b` al producto `WX`.

    **Pregunta 2.** ¿Cómo inicializa la función `initialize_parameters` los pesos y el sesgo? ¿Qué predicción inicial de regresión y qué probabilidad inicial de clasificación resultan si todos los pesos y el sesgo valen cero?
    """)
    return


@app.cell
def _(mo):
    # TODO 1: convierte los datos a las dimensiones requeridas por src.lib
    X_array = ...
    y_num_array = ...
    y_cat_array = ...

    mo.stop(any(value is ... for value in (X_array, y_num_array, y_cat_array)))

    # TODO 2: obtén n_x y n_y con layer_sizes e inicializa W y b
    n_x, n_y = ..., ...
    parameters_initial = ...

    mo.stop(parameters_initial is ...)
    print("X_array shape:", X_array.shape)
    print("W shape:", parameters_initial["W"].shape)
    print("b shape:", parameters_initial["b"].shape)
    return X_array, y_cat_array, y_num_array


@app.cell
def _(mo):
    mo.md(r"""
    **Respuestas 1 y 2.** Escribe aquí.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Una iteración de entrenamiento

    Usa las primeras cinco observaciones y los mismos parámetros iniciales para calcular la salida de regresión y de clasificación. Calcula el costo de cada una usando las funciones de `compute_cost_mse`y `compute_cost_bce`.

    Para regresión, llama a `backward_propagation_regression` y escribe las dos actualizaciones del descenso del gradiente. Usa los parámetros actualizados para calcular de nuevo el costo en esas cinco observaciones.

    **Pregunta 3.** ¿Qué calcula `forward_propagation` en cada tarea? ¿Qué función de costo corresponde a cada salida?

    **Pregunta 4.** ¿Cuáles son las dimensiones de `dW` y `db`? Al revisar las dos funciones de `backward_propagation`, ¿qué cambia en `dZ` (el gradiente) entre regresión y clasificación?

    **Pregunta 5.** ¿Cómo cambió el costo de regresión después de la actualización? Explica el papel del signo menos y de la tasa de aprendizaje.
    """)
    return


@app.cell
def _(X_array, mo, y_cat_array, y_num_array):
    X_step = X_array[:, :5]
    y_num_step = y_num_array[:, :5]
    y_cat_step = y_cat_array[:, :5]

    # TODO 3: calcula las salidas y los costos en las cinco observaciones
    y_hat_step = ...
    prob_step = ...
    cost_regression_step = ...
    cost_classification_step = ...

    mo.stop(any(value is ... for value in (
        y_hat_step, prob_step, cost_regression_step, cost_classification_step
    )))
    print("Costo de regresión:", cost_regression_step)
    print("Costo de clasificación:", cost_classification_step)
    return (cost_regression_step,)


@app.cell
def _(cost_regression_step, mo):
    learning_rate_step = 0.05

    # TODO 4: calcula los gradientes y aplica W - learning_rate * dW a W
    # Aplica la misma regla a b. Guarda los resultados en parameters_updated.
    grads_step = ...
    W_updated = ...
    b_updated = ...
    parameters_updated = {"W": W_updated, "b": b_updated}

    mo.stop(any(value is ... for value in (grads_step, W_updated, b_updated)))

    # TODO 5: vuelve a calcular el costo con los parámetros actualizados
    cost_updated = ...

    print("Costo antes de actualizar:", cost_regression_step)
    print("Costo después de actualizar:", cost_updated)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Respuestas 3, 4 y 5.** Escribe aquí.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Entrenamiento con todos los datos

    Entrena ambos perceptrones con las funciones de `nn_model_regression` y `nn_model_classification`, respectivamente. Prueba 1,000 iteraciones con tasa de aprendizaje de 0.2 para regresión y de 1.2 para clasificación. Usa `print_cost=False`.

    Genera las predicciones. Para clasificación, asigna clase 1 si la probabilidad supera 0.5, como en el notebook de repaso.
    """)
    return


@app.cell
def _(mo):
    # TODO 6: entrena los dos perceptrones
    parameters_regression = ...
    parameters_classification = ...

    mo.stop(parameters_regression is ... or parameters_classification is ...)

    # TODO 7: obtén las predicciones de regresión, probabilidades y clases
    y_pred_regression = ...
    prob_pred_classification = ...
    threshold = 0.5
    y_pred_classification = ...

    mo.stop(y_pred_regression is ... or y_pred_classification is ...)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Comparación con scikit-learn

    Entrena `LinearRegression` y `LogisticRegression` con las mismas entradas estandarizadas. Para comparar con la entropía cruzada binaria utilizada en este notebook, usa `LogisticRegression(max_iter=1000, C=np.inf)` para omitir la regularización.

    Calcula $R^2$ y MSE para ambos modelos de regresión. Para clasificación, calcula *accuracy* y F1 para ambos modelos. Completa las dos tablas.

    **Pregunta 6.** ¿Por qué la regresión lineal y la regresión logística pueden verse como casos particulares de un perceptrón de una capa? Explica qué comparten en cada comparación y por qué sus métricas podrían diferir.
    """)
    return


@app.cell
def _(mo, pd):
    # TODO 8: entrena los modelos de sklearn y genera predicciones
    model_regression = ...
    model_classification = ...
    y_pred_regression_sklearn = ...
    y_pred_classification_sklearn = ...

    mo.stop(y_pred_regression_sklearn is ... or y_pred_classification_sklearn is ...)

    # TODO 9: sustituye cada ... por la métrica correspondiente
    comparison_regression = pd.DataFrame({
        "Modelo": ["Perceptrón", "LinearRegression"],
        "R²": [..., ...],
        "MSE": [..., ...],
    })
    comparison_classification = pd.DataFrame({
        "Modelo": ["Perceptrón", "LogisticRegression"],
        "Accuracy": [..., ...],
        "F1": [..., ...],
    })

    mo.vstack([comparison_regression, comparison_classification])
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Respuesta 6.** Escribe aquí.
    """)
    return


@app.cell
def _(np):
    ## Funciones de python utilizadas para entrenamiento de un perceptron

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def layer_sizes(X,Y):
        """
        Argumentos:
        X -- conjunto de datos de entrada con dimensión (tamaño de entrada, número de ejemplos)
        Y -- etiquetas (valores objetivo) con dimensión (tamaño de salida, número de ejemplos)
    
        Retorna:
        n_x -- tamaño de la capa de entrada
        n_y -- tamaño de la capa de salida
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]
    
        return (n_x, n_y)

    def initialize_parameters(n_x, n_y):
        """
        Inicializa los parámetros del modelo (pesos y sesgos).

        Argumentos:
        n_x -- tamaño de la capa de entrada (número de características)
        n_y -- tamaño de la capa de salida

        Retorna:
        parameters -- diccionario de Python que contiene:
            W -- matriz de pesos de dimensión (n_y, n_x)
            b -- vector de sesgos de dimensión (n_y, 1)
        """
    
        # Inicialización aleatoria pequeña para evitar simetrías
        W = np.random.randn(n_y, n_x) * 0.01
    
        # Inicialización del sesgo en cero
        b = np.zeros((n_y, 1))
    
        parameters = {
            "W": W,
            "b": b
        }
    
        return parameters

    def forward_propagation_regression(X, parameters):
        """
        Implementa la propagación hacia adelante (forward propagation).

        Argumentos:
        X -- datos de entrada de dimensión (n_x, m)
        parameters -- diccionario de Python que contiene los parámetros del modelo

        Retorna:
        Y_hat -- salida del modelo (predicciones)
        """
    
        W = parameters["W"]
        b = parameters["b"]
    
        # Propagación hacia adelante: cálculo de Z
        Z = np.matmul(W, X) + b
    
        # En este caso (regresión lineal), la salida es identidad
        Y_hat = Z

        return Y_hat

    def forward_propagation_classification(X, parameters):
        """
        Implementa la propagación hacia adelante (forward propagation).

        Argumentos:
        X -- datos de entrada de dimensión (n_x, m)
        parameters -- diccionario de Python que contiene los parámetros del modelo

        Retorna:
        Y_hat -- salida del modelo (predicciones)
        """
    
        W = parameters["W"]
        b = parameters["b"]
    
        # Propagación hacia adelante: cálculo de Z
        Z = np.matmul(W, X) + b
    
        # En este caso (regresión lineal), la salida es identidad
        A = sigmoid(Z)

        return A

    def compute_cost_mse(Y_hat, Y):
        """
        Calcula la función de costo utilizando la suma de cuadrados.

        Argumentos:
        Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
        Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
        Retorna:
        cost -- valor de la función de costo, escalado por 1/(2*m)
        """
    
        # Número de ejemplos
        m = Y_hat.shape[1]

        # Cálculo de la función de costo
        cost = np.sum((Y_hat - Y)**2) / (2 * m)
    
        return cost

    def compute_cost_bce(A, Y):
        """
        Calcula la función de costo utilizando la entropía cruzada binaria.

        Argumentos:
        Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
        Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
        Retorna:
        cost -- valor de la función de costo, escalado por 1/(2*m)
        """
    
        # Número de ejemplos
        m = Y.shape[1]

        # Cálculo de la función de costo

        logprobs = (
            - np.multiply(np.log(A), Y)
            - np.multiply(np.log(1 - A), 1 - Y)
        )

        cost = 1 / m * np.sum(logprobs)
    
        return cost

    def backward_propagation_regression(Y_hat, X, Y):
        """
        Implementa la propagación hacia atrás (backpropagation), calculando los gradientes.

        Argumentos:
        Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
        X -- datos de entrada de dimensión (n_x, número de ejemplos)
        Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
        Retorna:
        grads -- diccionario de Python que contiene los gradientes respecto a los parámetros
        """
    
        m = X.shape[1]
    
        # Propagación hacia atrás: cálculo de derivadas parciales
        dZ = Y_hat - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    
        grads = {
            "dW": dW,
            "db": db
        }
    
        return grads

    def backward_propagation_classification(A, X, Y):
        """
        Implementa la propagación hacia atrás (backpropagation), calculando los gradientes.

        Argumentos:
        Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
        X -- datos de entrada de dimensión (n_x, número de ejemplos)
        Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
        Retorna:
        grads -- diccionario de Python que contiene los gradientes respecto a los parámetros
        """
    
        m = X.shape[1]
    
        # Error del modelo
        dZ = A - Y

        # Gradiente respecto a los pesos
        dW = 1 / m * np.dot(dZ, X.T)

        # Gradiente respecto al sesgo
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    
        grads = {
            "dW": dW,
            "db": db
        }
    
        return grads

    def update_parameters(parameters, grads, learning_rate=1.2):
        """
        Actualiza los parámetros utilizando la regla de descenso del gradiente.

        Argumentos:
        parameters -- diccionario de Python que contiene los parámetros del modelo
        grads -- diccionario de Python que contiene los gradientes
        learning_rate -- tasa de aprendizaje (learning rate)

        Retorna:
        parameters -- diccionario con los parámetros actualizados
        """
    
        # Extraer parámetros
        W = parameters["W"]
        b = parameters["b"]
    
        # Extraer gradientes
        dW = grads["dW"]
        db = grads["db"]
    
        # Regla de actualización (gradient descent)
        W = W - learning_rate * dW
        b = b - learning_rate * db
    
        parameters = {
            "W": W,
            "b": b
        }
    
        return parameters

    def nn_model_regression(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
        """
        Construye y entrena un modelo de red neuronal simple.

        Argumentos:
        X -- conjunto de datos de dimensión (n_x, número de ejemplos)
        Y -- etiquetas o valores reales de dimensión (n_y, número de ejemplos)
        num_iterations -- número de iteraciones del ciclo de entrenamiento
        learning_rate -- tasa de aprendizaje para el descenso del gradiente
        print_cost -- si es True, imprime el costo en cada iteración

        Retorna:
        parameters -- parámetros aprendidos por el modelo. Posteriormente pueden usarse para hacer predicciones.
        """
    
        n_x = layer_sizes(X, Y)[0]
        n_y = layer_sizes(X, Y)[1]
    
        parameters = initialize_parameters(n_x, n_y)
    
        # Ciclo de entrenamiento
        for i in range(0, num_iterations):
         
            # Propagación hacia adelante.
            # Entradas: X, parameters. Salida: Y_hat.
            Y_hat = forward_propagation_regression(X, parameters)
        
            # Función de costo.
            # Entradas: Y_hat, Y. Salida: cost.
            cost = compute_cost_mse(Y_hat, Y)
        
            # Propagación hacia atrás.
            # Entradas: Y_hat, X, Y. Salida: grads.
            grads = backward_propagation_regression(Y_hat, X, Y)
    
            # Actualización de parámetros mediante descenso del gradiente.
            # Entradas: parameters, grads, learning_rate. Salida: parameters.
            parameters = update_parameters(parameters, grads, learning_rate)
        
            # Imprimir el costo en cada iteración
            if print_cost:
                print("Costo después de la iteración %i: %f" % (i, cost))

        return parameters

    def nn_model_classification(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
        """
        Argumentos:
        X -- conjunto de datos con forma
             (n_x, número de ejemplos)

        Y -- etiquetas con forma
             (n_y, número de ejemplos)

        num_iterations -- número de iteraciones del entrenamiento

        learning_rate -- tasa de aprendizaje utilizada
                         en descenso del gradiente

        print_cost -- si es True, imprime el costo
                      en cada iteración

        Retorna:
        parameters -- parámetros aprendidos por el modelo.
                      Posteriormente pueden utilizarse
                      para hacer predicciones.
        """

        # Obtener tamaños de entrada y salida
        n_x = layer_sizes(X, Y)[0]
        n_y = layer_sizes(X, Y)[1]

        # Inicializar parámetros
        parameters = initialize_parameters(n_x, n_y)

        # Loop de entrenamiento

        for i in range(0, num_iterations):

            # Forward propagation

            A = forward_propagation_classification(X, parameters)

            # Cálculo del costo

            cost = compute_cost_bce(A, Y)

            # Back propagation

            grads = backward_propagation_classification(A, X, Y)

            # Actualización de parámetros

            parameters = update_parameters(
                parameters,
                grads,
                learning_rate
            )

            # Imprimir costo en cada iteración
            if print_cost:
                print("Costo después de la iteración %i: %f" % (i, cost))

        return parameters

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
