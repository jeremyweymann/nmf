'use strict';

const {Matrix} = require('ml-matrix');


module.exports = augmentedMatrix;

/**
 * Computes the augmented matrix for regularization of one of the matrix 
 * @param {Matrix} X 
 * @param {Matrix} Y 
 * @param {number} k 
 * @param {number} sqrtSparseParam 
 * @param {number} sqrtScalingParam 
 */

function augmentedMatrix(X, Y, k, sqrtSparseParam, sqrtScalingParam) {
    let rowAugmentedX = vertConcatMatrix(X, Matrix.ones(1, k).mul(sqrtSparseParam));
    let idAugmentedYt = vertConcatMatrix(Y.transpose(), Matrix.eye(k).mul(sqrtScalingParam));

    return {rowAugmentedX, idAugmentedYt};
}


/**
 * 
 * @param {Matrix} A 
 * @param {Matrix} B 
 */

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
