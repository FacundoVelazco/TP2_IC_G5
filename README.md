# TP2_IC_G5
Trabajo practico 2 de Inteligencia computacional, grupo 5.

## Idea global

Configurar un algoritmo genético que optimice hiperparámetros de una red neuronal.

### Idea inicial

Trabajar con un cromosoma simple, como podria ser
* Cantidad de capas ocultas {3}.
* Cantidad de neuronas por capa {x,y,z}.
* Funcion de activación de la capa {f1,f2,f3}.

### Idea avanzada

Combinar los tres cromosomas anteriores en un Framework que genere configuraciones de hiperparametros principales de redes neuronales.
Esto se realizaría formando una jerarquia

![iamge.png](https://cloud.google.com/static/vertex-ai/docs/tabular-data/forecasting/images/hierarchy.svg?hl=es-419
) 

Imagino que se podria aplicar recursivamente GA en cada nivel de la jerarquia. Formando asi un cromosoma compuesto {a,{x,y}.{f1,f2}}