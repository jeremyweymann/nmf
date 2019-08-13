'use strict';

const {Matrix} = require('ml-matrix');
const {fcnnls} = require('fcnnls');

module.exports = anlsFcnnls;

function anlsFcnnls(V, k, options = {}) {

    const m = V.rows;
    const n = V.columns;
    let maxV = V.max();
    const {
        Winit = Matrix.rand(m, k).mul(Math.sqrt(maxV)),
        Hinit = Matrix.rand(k, n),
        tol = 0.000000001,
        maxIter = 1000,
    } = options;

    let iter = 0;

    let Wt;
    let W = Winit;
    let H = Hinit;
    let relativeError;

    let convCrit;

    let delta0 = Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2));
    do {
        iter++;
        H = fcnnls(W, V);
        Wt = fcnnls(H.transpose(), V.transpose());
        W = Wt.transpose();
        convCrit = (Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;

    } while (convCrit > tol && iter < maxIter);

    relativeError = Matrix.subtract(Wt.transpose().mmul(H), V).norm() / V.norm();

    console.log({iter, relativeError, convCrit});
    //console.log('\nIter = '+i);
    return {W: W, H: H};
}


function projGradCostFunc(A, X, Y) {
    let C = X.mmul(Y).mmul(Y.transpose());
    let D = A.mmul(Y.transpose());
    let projGrad = Matrix.sub(C, D);
    //let projGrad = Matrix.sub(X.mmul(Y).mmul(Y.transpose()), A.mmul(Y.transpose())).mul(2);
    for (let i = 0; i < projGrad.rows; i++) {
        for (let j = 0; j < projGrad.columns; j++) {
            if (projGrad.get(i, j) >= 0 && X.get(i, j) <= 0) projGrad.set(i, j, 0);
        }
    }
    return projGrad;
}

