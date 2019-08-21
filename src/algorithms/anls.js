'use strict';

const {Matrix} = require('ml-matrix');

const projGrad = require('../util/projGrad');
const {augmentedMatrix, vertConcatMatrix} = require('../util/augmentedMatrix');

module.exports = anls;

function anls(V, k, method, Winit, Hinit, options = {}) {

    const m = V.rows;
    const n = V.columns;
    let maxV = V.max();

    const {
        tol = 0.000000001,
        maxIter = 2000,
        regularization = 0,
        sparseParam = 0.01,
        scalingParam = maxV,
    } = options;

    let W = Winit;
    let H = Hinit;

    let relativeError;
    let convCrit;
    let iter = 0;


    if (regularization === 'R') {

        let sqrtSparseParam = Math.sqrt(sparseParam);
        let sqrtScalingParam = Math.sqrt(scalingParam);

        let delta0 = Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2));

        let rowAugmentedV = vertConcatMatrix(V, Matrix.zeros(1, n));
        let matAugmentedVt = vertConcatMatrix(V.transpose(), Matrix.zeros(k, m));

        do {
            iter++;
            let augmentedMat = augmentedMatrix(W, H, k, sqrtSparseParam, sqrtScalingParam);
            let rowAugmentedW = augmentedMat.rowAugmentedX;
            let idAugmentedHt = augmentedMat.idAugmentedYt;

            H = method(rowAugmentedW, rowAugmentedV);
            W = method(idAugmentedHt, matAugmentedVt).transpose();
            convCrit = (Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;
        } while (convCrit > tol && iter < maxIter);

    } else if (regularization === 'L') {

        let sqrtSparseParam = Math.sqrt(sparseParam);
        let sqrtScalingParam = Math.sqrt(scalingParam);

        let delta0 = Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2));


        let rowAugmentedVt = vertConcatMatrix(V.transpose(), Matrix.zeros(1, m));
        let matAugmentedV = vertConcatMatrix(V, Matrix.zeros(k, n));

        do {
            iter++;
            let augmentedMat = augmentedMatrix(H.transpose(), W.transpose(), k, sqrtSparseParam, sqrtScalingParam);
            let rowAugmentedHt = augmentedMat.rowAugmentedX;
            let idAugmentedW = augmentedMat.idAugmentedYt;

            W = method(rowAugmentedHt, rowAugmentedVt).transpose();
            H = method(idAugmentedW, matAugmentedV);
            convCrit = (Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;
        } while (convCrit > tol && iter < maxIter);

    } else {
        //console.log({W, H});

        let delta0 = Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2));
        do {
            iter++;
            H = method(W, V);
            W = method(H.transpose(), V.transpose()).transpose();
            convCrit = (Math.sqrt(Math.pow(projGrad(V, W, H).norm(), 2) + Math.pow(projGrad(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;
        } while (convCrit > tol && iter < maxIter);

    }

    relativeError = Matrix.subtract(W.mmul(H), V).norm() / V.norm();

    console.log({iter, relativeError, convCrit});
    //console.log('\nIter = '+i);
    return {W: W, H: H};
}

