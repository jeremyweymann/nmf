'use strict';

const {Matrix} = require('ml-matrix');

module.exports = pgdnnls;

function pgdnnls(X, A, options = {}) {
    const {maxIterations = 2, Yinit = Matrix.rand(X.columns, A.columns)} = options;

    let Y = Yinit;
    let XtA = X.transpose().mmul(A);
    let XtX = X.transpose().mmul(X);

    let grad;
    let alpha = 1;
    let beta = 0.1;
    let decrAlpha;
    let Yp;

    //let numberIterations;

    for (let iter = 1; iter < maxIterations; iter++) {
        //numberIterations = iter;
        grad = Matrix.subtract(XtX.mmul(Y), XtA);
        //let projgrad = norm2(selectElementsFromMatrix(grad, logicalOrMatrix(elementsMatrixInferiorZero(grad), elementsMatrixSuperiorZero(H))));
        for (let innerIter = 1; innerIter < 20; innerIter++) {
            let Yn = Matrix.sub(Y, Matrix.mul(grad, alpha));
            Yn = replaceElementsMatrix(Yn, elementsMatrixSuperiorZero(Yn), 0);
            let d = Matrix.sub(Yn, Y);
            let gradd = (d.mul(grad)).sum();
            let dQd = (XtX.mmul(d).mul(d)).sum();
            let suffDecr = 0.99 * gradd + 0.5 * dQd < 0;
            if (innerIter === 1) {
                decrAlpha = !suffDecr;
                Yp = Y.clone();
            }
            if (decrAlpha) {
                if (suffDecr) {
                    Y = Yn.clone();
                    break;
                } else {
                    alpha = alpha * beta;
                }
            } else {
                if (!suffDecr || matrixEqual(Y, Yp)) {
                    Y = Yp.clone();
                    break;
                } else {
                    alpha = alpha / beta;
                    Yp = Y.clone();
                }
            }
        }

        /*if (iter === maxIter) {
            console.log('Max iterations in nlssubprob');
        }*/
    }
    return Y;
}


/**
 * @private
 * Take a matrix, a 2D-array of booleans (same dimensions than the matrix) and a value. Replace the elements of the matrix which corresponds to a value true in the 2D-array of booleans by the value given into parameter. Return a matrix.
 * @param {Matrix} X
 * @param {Array<Array<boolean>>} arrayBooleans
 * @param {number} value
 * @return {Matrix} Matrix which the replaced elements.
 */

function replaceElementsMatrix(X, arrayBooleans, value) {
    if (
        X.rows !== arrayBooleans.length ||
        X.columns !== arrayBooleans[0].length
    ) {
        throw new Error('Error of dimension');
    }
    let rows = X.rows;
    let columns = X.columns;
    let newMatrix = new Matrix(X);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            if (!arrayBooleans[r][c]) {
                newMatrix.set(r, c, value);
            }
        }
    }
    return newMatrix;
}


/**
 * @private
 * Return if two matrix are equals (i.e each element is equal to the corresponding element of the other matrix).
 * @param {Matrix} m1
 * @param {Matrix} m2
 * @return {boolean}
 */

function matrixEqual(m1, m2) {
    if (m1.rows !== m2.rows || m1.columns !== m2.columns) {
        throw new Error('Error of dimension');
    }
    let rows = m1.rows;
    let columns = m1.columns;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < columns; c++) {
            if (m1.get(r, c) !== m2.get(r, c)) {
                return false;
            }
        }
    }
    return true;
}


/**
 * @private
 * Return a 1D-Array with the elements of the matrix X which are superior than 0
 * @param {Matrix} X
 * @return {Array<number>} elements superior than 0
 */

function elementsMatrixSuperiorZero(X) {
    let newArray = new Array(X.rows);
    for (let i = 0; i < newArray.length; i++) {
        newArray[i] = new Array(X.columns);
        for (let j = 0; j < X.columns; j++) {
            newArray[i][j] = X.get(i, j) > 0;
        }
    }
    return newArray;
}

