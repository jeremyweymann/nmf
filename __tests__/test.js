'use strict';

const nmf = require('../src');
const gradientAdditive = require('../src/algorithms/gradientAdditive');
const anlsFcnnls = require('../src/algorithms/anlsFcnnls');

const {Matrix} = require('ml-matrix');
const {toBeDeepCloseTo} = require('jest-matcher-deep-close-to');
expect.extend({toBeDeepCloseTo});

describe('NMF test', () => {
    it.skip('use case 1', async () => {
        let W = new Matrix([[1, 2, 3], [4, 5, 6]]);
        let H = new Matrix([[1, 2], [3, 4], [5, 6]]);
        let Winit = new Matrix([[1, 1, 3], [4, 5, 6]]);
        let Hinit = new Matrix([[1, 1], [3, 4], [5, 6]]);

        let V = W.mmul(H);

        const options = {
            Winit: Winit,
            Hinit: Hinit,
            tol: 0.001,
            maxIter: 10
        };

        const result = nmf(V, 2, gradientAdditive, options);
        const w0 = result.W;
        const h0 = result.H;
        expect(w0.mmul(h0).to2DArray()).toBeDeepCloseTo(V.to2DArray(), 1);
    });

    it('Test ANLS-FCNNLS', async () => {

        //let V = new Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
        let V = Matrix.randInt(100, 20);

        let maxV = V.max();

        const nmfResult = nmf(V, 4, anlsFcnnls);
        let result = (nmfResult.W).mmul(nmfResult.H);
        console.log({V, result, maxV});

        expect(result.to2DArray()).toBeDeepCloseTo(V.to2DArray());
    });
});
