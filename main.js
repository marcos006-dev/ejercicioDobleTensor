import * as tf from '@tensorflow/tfjs';
const btnCalcular = document.getElementById('btnCalcular');
const estadoModelo = document.getElementById('estadoModelo');
const numeroEscalar = document.getElementById('numeroEscalar');

let modelo;

document.addEventListener('DOMContentLoaded', async () => {
  estadoModelo.innerHTML = `
  <div class="alert alert-warning" role="alert">
    Entrenando el modelo...
  </div>
  `;

  // declarar constante para el grafico
  const chart = new Highcharts.chart('funcionPerdida', {
    title: {
      text: 'Datos de la función de perdida',
    },
    xAxis: {
      categories: 0,
    },
    series: [
      {
        name: 'Datos de la función de perdida',
        data: 0,
      },
    ],
    credits: {
      enabled: false,
    },
  });
  // definir el modelo
  modelo = tf.sequential();

  // definir las capas

  modelo.add(
    tf.layers.dense({ inputShape: [1], units: 6, activation: 'relu' })
  );
  modelo.add(tf.layers.dense({ units: 3 }));
  // modelo.add(tf.layers.dense({ units: 1 }));
  // modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // compilar el modelo

  modelo.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // definir los datos de entrenamiento (tensores)

  const variableY = tf.tensor1d([1, 2, 3, 4], 'int32');
  const variableX = tf.tensor1d([2, 4, 6, 8], 'int32');

  // entrenar el modelo
  await modelo.fit(variableY, variableX, {
    epochs: 500,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        chart.series[0].addPoint(logs.loss);
      },
    },
  });

  estadoModelo.innerHTML = `
  <div class="alert alert-success" role="alert">
    Modelo Entrenado
  </div>
  `;

  btnCalcular.removeAttribute('disabled');
  numeroEscalar.removeAttribute('disabled');
});

btnCalcular.addEventListener('click', async () => {
  const numero = parseInt(numeroEscalar.value);

  // modelo.predict(tf.tensor1d([numero])).print();

  estadoModelo.innerHTML = `
  <div class="alert alert-info" role="alert">
    Predicción del modelo para el número ${numero} =>
    ${modelo.predict(tf.tensor1d([numero])).dataSync()}
  </div>
  `;
});
// resultados del modelo ingresados con los valores de X
// Tensor
//[[4.0038075],
//[7.9959173],
//[11.9880247]]

// valores de x sin reemplazar
// [2,4,6]

// valores de x reemplazados con los datos arrojados del modelo
// [4,8,12]
/*Insertamos los valores del eje vertical en un array*/
