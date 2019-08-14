'use strict';

const {Matrix} = require('ml-matrix');
const {fcnnls} = require('fcnnls');

module.exports = anlsActiveSet;

/**
 * 
 * @param {*} V 
 * @param {*} k 
 * @param {*} options include: Winit, Hinit, tol, maxIter, regularisation (W, H or both) 
 */

function anlsActiveSet(V, k, options = {}) {

    const m = V.rows;
    const n = V.columns;
    let maxV = V.max();

    const {
        Winit = Matrix.rand(m, k).mul(Math.sqrt(maxV)),
        Hinit = Matrix.rand(k, n).mul(Math.sqrt(maxV)),
        tol = 0.000000001,
        maxIter = 1000,
        regularization = 0,
        sparseParam = 0.01,
        scalingParam = maxV,
    } = options;

    let iter = 0;

    let W = Winit;
    let H = Hinit;
    let relativeError;

    let convCrit;


    // AugmentedV 

    if (regularization === 'R') {
        let sqrtSparseParam = Math.sqrt(sparseParam);
        let sqrtScalingParam = Math.sqrt(scalingParam);

        let delta0 = Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2));

        let rowAugmentedV = vertConcatMatrix(V, Matrix.zeros(1, n));
        let matAugmentedVt = vertConcatMatrix(V.transpose(), Matrix.zeros(k, m));

        do {
            iter++;
            let augmentedMat = augmentedMatrix(W, H, k, sqrtSparseParam, sqrtScalingParam);
            let rowAugmentedW = augmentedMat.rowAugmentedX;
            let idAugmentedHt = augmentedMat.idAugmentedYt;

            H = fcnnls(rowAugmentedW, rowAugmentedV);
            W = fcnnls(idAugmentedHt, matAugmentedVt).transpose();
            convCrit = (Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;

        } while (convCrit > tol && iter < maxIter);


    } else if (regularization === 'L') {

        let sqrtSparseParam = Math.sqrt(sparseParam);
        let sqrtScalingParam = Math.sqrt(scalingParam);

        let delta0 = Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2));


        let rowAugmentedVt = vertConcatMatrix(V.transpose(), Matrix.zeros(1, m));
        let matAugmentedV = vertConcatMatrix(V, Matrix.zeros(k, n));

        do {
            iter++;
            let augmentedMat = augmentedMatrix(H.transpose(), W.transpose(), k, sqrtSparseParam, sqrtScalingParam);
            let rowAugmentedHt = augmentedMat.rowAugmentedX;
            let idAugmentedW = augmentedMat.idAugmentedYt;


            W = fcnnls(rowAugmentedHt, rowAugmentedVt).transpose();
            H = fcnnls(idAugmentedW, matAugmentedV);
            convCrit = (Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;

        } while (convCrit > tol && iter < maxIter);


    } else {


        let delta0 = Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2));
        do {
            iter++;
            H = fcnnls(W, V);
            W = fcnnls(H.transpose(), V.transpose()).transpose();
            convCrit = (Math.sqrt(Math.pow(projGradCostFunc(V, W, H).norm(), 2) + Math.pow(projGradCostFunc(V.transpose(), H.transpose(), W.transpose()).norm(), 2))) / delta0;

        } while (convCrit > tol && iter < maxIter);

    }

    relativeError = Matrix.subtract(W.mmul(H), V).norm() / V.norm();

    console.log({iter, relativeError, convCrit});
    //console.log('\nIter = '+i);
    return {W: W, H: H};
}


function projGradCostFunc(A, X, Y) {
    let projGrad = Matrix.sub(X.mmul(Y).mmul(Y.transpose()), A.mmul(Y.transpose())).mul(2);
    for (let i = 0; i < projGrad.rows; i++) {
        for (let j = 0; j < projGrad.columns; j++) {
            if (projGrad.get(i, j) >= 0 && X.get(i, j) <= 0) projGrad.set(i, j, 0);
        }
    }

    return projGrad;
}

function augmentedMatrix(X, Y, k, sqrtSparseParam, sqrtScalingParam) {
    let rowAugmentedX = vertConcatMatrix(X, Matrix.ones(1, k).mul(sqrtSparseParam));
    let idAugmentedYt = vertConcatMatrix(Y.transpose(), Matrix.eye(k).mul(sqrtScalingParam));

    return {rowAugmentedX, idAugmentedYt};
}

function vertConcatMatrix(A, B) {
    if (A.columns !== B.columns) {
        throw new Error('Matrices dimensions must agree');
    }

    let m = A.rows + B.rows;
    let n = A.columns;
    let C = Matrix.zeros(m, n);

    C.setSubMatrix(A, 0, 0);
    C.setSubMatrix(B, A.rows, 0);

    return C;
}
