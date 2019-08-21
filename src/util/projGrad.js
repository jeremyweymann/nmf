'use strict';

const {Matrix, WrapperMatrix2D} = require('ml-matrix');

module.exports = projGrad;

/**
 * Computes projected gradient of the cost function ||A - XY||^2 (euclidean distance) for given A, X and Y for the stopping criterion.
 * @link https://epubs.siam.org/doi/abs/10.1137/110821172
 * @param {Matrix} A 
 * @param {Matrix} X 
 * @param {Matrix} Y 
 */

function projGrad(A, X, Y) {
    A = WrapperMatrix2D.checkMatrix(A);
    X = WrapperMatrix2D.checkMatrix(X);
    Y = WrapperMatrix2D.checkMatrix(Y);

    if (X.columns !== Y.rows || X.rows !== A.rows || Y.columns !== A.columns) {
        throw new Error('Matrices dimensions must agree');
    }

    let projGrad = Matrix.sub(X.mmul(Y).mmul(Y.transpose()), A.mmul(Y.transpose())).mul(2);
    for (let i = 0; i < projGrad.rows; i++) {
        for (let j = 0; j < projGrad.columns; j++) {
            if (projGrad.get(i, j) >= 0 && X.get(i, j) <= 0) projGrad.set(i, j, 0);
        }
    }

    return projGrad;
}
