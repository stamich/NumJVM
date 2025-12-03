package org.numjvm.codegen;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * One generated operation: its name, arity, output dtype rule and, per dtype family, the
 * Java expression computing one element. Binary expressions use {@code x} and {@code y},
 * unary ones {@code x}; both are values of the family's compute type. A family without
 * an expression gets no loop.
 *
 * @param name        operation name, as in NumPy
 * @param doc         short description for the generated Javadoc
 * @param arity       1 or 2
 * @param boolOutput  whether the output is bool (comparisons, logical operations)
 *                    rather than the input dtype
 * @param expressions expression per family
 * @param only        if not empty, generate loops only for these dtypes
 */
record OpSpec(String name, String doc, int arity, boolean boolOutput, Map<DTypeSpec.Family, String> expressions,
              List<String> only) {

    String expression(DTypeSpec t) {
        if (!only.isEmpty() && !only.contains(t.dtype())) {
            return null;
        }
        return expressions.get(t.family());
    }

    DTypeSpec output(DTypeSpec in) {
        return boolOutput ? DTypeSpec.bool() : in;
    }

    /**
     * Builds an operation spec.
     */
    static final class Builder {
        private final String name;
        private final String doc;
        private final int arity;
        private boolean boolOutput;
        private final Map<DTypeSpec.Family, String> expressions = new EnumMap<>(DTypeSpec.Family.class);
        private final List<String> only = new ArrayList<>();

        Builder(String name, int arity, String doc) {
            this.name = name;
            this.arity = arity;
            this.doc = doc;
        }

        Builder boolOutput() {
            boolOutput = true;
            return this;
        }

        Builder on(String expression, DTypeSpec.Family... families) {
            for (DTypeSpec.Family f : families) {
                expressions.put(f, expression);
            }
            return this;
        }

        Builder only(String... dtypes) {
            only.addAll(List.of(dtypes));
            return this;
        }

        OpSpec build() {
            return new OpSpec(name, doc, arity, boolOutput, Map.copyOf(expressions), List.copyOf(only));
        }
    }

    static Builder binary(String name, String doc) {
        return new Builder(name, 2, doc);
    }

    static Builder unary(String name, String doc) {
        return new Builder(name, 1, doc);
    }
}
