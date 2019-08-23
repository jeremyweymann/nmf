'use strict';

const {Matrix} = require('ml-matrix');

module.exports = muEuclidean;

function muEuclidean(V, W, H, k, numberIterations = 1000) {
    let numW = Matrix.empty(V.rows, k);
    let denumW = Matrix.empty(V.rows, k);
    let numH = Matrix.empty(k, V.columns);
    let denumH = Matrix.empty(k, V.columns);

    for (let a = 0; a < numberIterations; a++) {
        numW = V.mmul(H.transpose());
        denumW = (W.mmul(H)).mmul(H.transpose());
        numW = numW.add(Number.EPSILON);
        denumW = denumW.add(Number.EPSILON);
        W = W.mul(numW.divide(denumW));

        numH = W.transpose().mmul(V);
        denumH = (W.transpose().mmul(W)).mmul(H);
        numH = numH.add(Number.EPSILON);
        denumH = denumH.add(Number.EPSILON);
        H = H.mul(numH.divide(denumH));
    }
}
