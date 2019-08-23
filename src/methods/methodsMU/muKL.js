'use strict';

const {Matrix} = require('ml-matrix');

module.exports = muKL;

function muKL(V, W, H, k, numberIterations = 1000) {
    let WH = W.mmul(H);

    let numW = Matrix.empty(V.rows, k);
    let denumW = Matrix.empty(V.rows, k);

    let numH = Matrix.empty(k, V.columns);
    let denumH = Matrix.empty(k, V.columns);

    let temp1 = Matrix.empty(V.rows, V.columns);
    let temp2 = Matrix.empty(V.columns, V.rows);

    for (let a = 0; a < numberIterations; a++) {
        for (let i = 0; i < V.rows; i++) {
            for (let j = 0; j < V.columns; j++) {
                if (WH.get(i, j) === 0) {
                    temp1.set(i, j, 0);
                    temp2.set(i, j, 0);
                } else {
                    temp1.set(i, j, Math.pow(WH.get(i, j), -2) * V.get(i, j));
                    temp2.set(i, j, Math.pow(WH.get(i, j), -1));
                }
            }
        }

        numW = temp1.mmul(H.transpose());
        denumW = temp2.mmul(H.transpose());
        numW = numW.add(Number.EPSILON);
        denumW = denumW.add(Number.EPSILON);
        W = W.mul(numW.divide(denumW));

        WH = W.mmul(H).add(Number.EPSILON);

        for (let i = 0; i < V.rows; i++) {
            for (let j = 0; j < V.columns; j++) {
                temp1.set(i, j, Math.pow(WH.get(i, j), -2) * V.get(i, j));
                temp2.set(i, j, Math.pow(WH.get(i, j), -1));
            }
        }

        numH = W.transpose().mmul(temp1);
        denumH = W.transpose().mmul(temp2);
        numH = numH.add(Number.EPSILON);
        denumH = denumH.add(Number.EPSILON);
        H = H.mul(numH.divide(denumH));

        WH = W.mmul(H).add(Number.EPSILON);
    }
}
