package org.numjvm.vector;

import jdk.incubator.vector.DoubleVector;

/** Implemented by one hidden class generated per supported graph. */
interface DoubleExpression {
    DoubleVector evaluate(DoubleVector[] inputs);
}
