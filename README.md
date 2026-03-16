# Matches vs Inliers en ORB + RANSAC

Este documento explica la diferencia entre **matches** e **inliers**
usando el mismo flujo de procesamiento utilizado en el código del
proyecto.

El objetivo es comparar dos imágenes utilizando **ORB features**,
encontrar correspondencias y luego filtrar coincidencias incorrectas
usando **RANSAC**.

------------------------------------------------------------------------

# 1. Match

Un **match** es una coincidencia entre descriptores de dos imágenes.

En el código esto ocurre en la siguiente línea:

``` python
matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
```

Cada match indica que:

-   Un **keypoint** detectado en la **imagen 1** tiene un descriptor
    similar a un **keypoint** en la **imagen 2**.


Esto significa que el algoritmo cree que esos puntos representan la
misma región de la escena.

Sin embargo, **no todos los matches son correctos**.

Pueden existir errores debido a:

-   texturas repetidas
-   ruido
-   iluminación diferente
-   patrones visuales similares

Por lo tanto, los matches iniciales pueden contener errores.

------------------------------------------------------------------------

# 2. Inlier

Un **inlier** es un match que **sí respeta la geometría de la escena**.

Para determinar esto usamos **RANSAC**.

En el código:

``` python
model_robust, inliers = ransac(
    (dst, src),
    AffineTransform,
    min_samples=4,
    residual_threshold=2,
    max_trials=1000
)
```

RANSAC funciona de la siguiente manera:

1.  Intenta calcular una transformación geométrica entre las imágenes.
2.  Evalúa qué matches son consistentes con esa transformación.

Los matches se clasifican en:

  Tipo      Significado
  --------- ------------------------------------
  Inlier    match consistente con la geometría
  Outlier   match incorrecto


Los **inliers representan correspondencias confiables**.


------------------------------------------------------------------------

# 3. Ejemplo con resultados del programa

Supongamos que el programa imprime:

    Matches: 188
    Inliers: 142
    Similarity score: 0.284

Interpretación:

  Métrica       Significado
  ------------- -----------------------------------------
  188 matches   coincidencias visuales entre imágenes
  142 inliers   coincidencias geométricamente correctas
  46 outliers   coincidencias incorrectas descartadas
