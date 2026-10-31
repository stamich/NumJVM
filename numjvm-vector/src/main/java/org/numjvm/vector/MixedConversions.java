package org.numjvm.vector;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/** Equal logical lane counts make every float-to-double transition explicit. */
final class MixedConversions {
    static final VectorSpecies<Double> DOUBLE = DoubleVector.SPECIES_PREFERRED;
    static final VectorSpecies<Float> FLOAT = DOUBLE.vectorBitSize() < 128 ? null
            : VectorSpecies.of(float.class, VectorShape.forBitSize(DOUBLE.vectorBitSize() / 2));
    private MixedConversions() { }
    static DoubleVector toDouble(FloatVector value) {
        return (DoubleVector) value.convertShape(VectorOperators.F2D, DOUBLE, 0);
    }
    static FloatVector toFloat(DoubleVector value) {
        return (FloatVector) value.convertShape(VectorOperators.D2F, FLOAT, 0);
    }
}
