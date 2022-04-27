import * as tf from '@tensorflow/tfjs';

// definir el modelo

const modelo = tf.sequential();

// definir las capas

modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));
modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// compilar el modelo

modelo.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// definir los datos de entrenamiento (tensores)

const variableY = tf.tensor1d([1, 3, 5], 'int32');
const variableX = tf.tensor1d([2, 4, 6], 'int32');

// entrenar el modelo

modelo.fit(variableY, variableX, { epochs: 500 }).then(() => {
  // predecir

  modelo.predict(tf.tensor1d([1, 2, 3, 6, 5, 10])).print();
});

// resultados del modelo ingresados con los valores de Y
// [[1.9615195 ],
//  [2.975291  ],
//  [3.9890623 ],
//  [7.0303764 ],
//  [6.0166049 ],
//  [11.0854626]]

// valores de x sin reemplazar
// [2, ?, 4, ?, 6, ?]

// valores de x reemplazados con los datos arrojados del modelo
// [2,3,4,7,6,11]
