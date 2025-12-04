package org.numjvm.internal.loops;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.numjvm.dtype.DType;
import org.numjvm.exception.DTypeException;

/**
 * All inner loops, keyed by operation and input dtypes.
 *
 * <p>Loops are registered once, while this class is initialised, by the classes that
 * implement them; afterwards the registry is immutable and safe to read from any thread.
 * For every operation the registry also keeps the order in which loops were registered:
 * type resolution walks that order to find the first loop an input can be cast to, as
 * NumPy does with a ufunc's {@code types} list.
 */
public final class KernelRegistry {

    private static final Map<String, Table> TABLES = build();

    private KernelRegistry() {
    }

    private static Map<String, Table> build() {
        Registrar r = new Registrar();
        GeneratedLoops.register(r);
        ComplexLoops.register(r);
        Map<String, Table> tables = new LinkedHashMap<>();
        r.tables.forEach((op, t) -> tables.put(op, t.freeze()));
        return Collections.unmodifiableMap(tables);
    }

    /**
     * Collects loops during registry initialisation. Passed to each loop class's
     * {@code register} method.
     */
    public static final class Registrar {

        private final Map<String, Table> tables = new LinkedHashMap<>();

        Registrar() {
        }

        private Table table(String op) {
            return tables.computeIfAbsent(op, Table::new);
        }

        /**
         * Registers a loop with one input.
         *
         * @param op   operation name
         * @param in   input dtype
         * @param out  output dtype
         * @param loop the loop
         */
        public void unary(String op, DType in, DType out, UnaryLoop loop) {
            table(op).add(new LoopSignature(op, List.of(in), out), loop);
        }

        /**
         * Registers a loop with two inputs.
         *
         * @param op   operation name
         * @param in1  first input dtype
         * @param in2  second input dtype
         * @param out  output dtype
         * @param loop the loop
         */
        public void binary(String op, DType in1, DType in2, DType out, BinaryLoop loop) {
            table(op).add(new LoopSignature(op, List.of(in1, in2), out), loop);
        }
    }

    /**
     * The loops of one operation.
     */
    private static final class Table {
        private final String op;
        private final List<LoopSignature> order = new ArrayList<>();
        private final Map<Integer, Object> loops = new HashMap<>();
        private final Map<Integer, LoopSignature> signatures = new HashMap<>();

        Table(String op) {
            this.op = op;
        }

        void add(LoopSignature sig, Object loop) {
            if (!order.isEmpty() && order.getFirst().arity() != sig.arity()) {
                throw new IllegalStateException("operation " + op + " registered with different arities");
            }
            int key = key(sig.inputs());
            if (loops.putIfAbsent(key, loop) != null) {
                throw new IllegalStateException("duplicate loop " + sig);
            }
            signatures.put(key, sig);
            order.add(sig);
        }

        Table freeze() {
            Table t = new Table(op);
            t.order.addAll(order);
            t.loops.putAll(loops);
            t.signatures.putAll(signatures);
            return t;
        }
    }

    private static int key(List<DType> inputs) {
        int k = 0;
        for (DType d : inputs) {
            k = k * 32 + d.ordinal() + 1;
        }
        return k;
    }

    private static int key(DType a) {
        return a.ordinal() + 1;
    }

    private static int key(DType a, DType b) {
        return (a.ordinal() + 1) * 32 + b.ordinal() + 1;
    }

    private static Table table(String op) {
        Table t = TABLES.get(op);
        if (t == null) {
            throw new DTypeException("no loops are registered for operation '" + op + "'");
        }
        return t;
    }

    /**
     * Returns the names of all operations with registered loops.
     *
     * @return operation names, in registration order
     */
    public static Set<String> operations() {
        return TABLES.keySet();
    }

    /**
     * Returns the signatures of an operation's loops, in registration order.
     *
     * @param op operation name
     * @return the signatures
     * @throws DTypeException if the operation has no loops
     */
    public static List<LoopSignature> signatures(String op) {
        return Collections.unmodifiableList(table(op).order);
    }

    /**
     * Returns the signature of the loop for exactly these input dtypes, or {@code null}.
     *
     * @param op     operation name
     * @param inputs input dtypes
     * @return the signature, or {@code null} if there is no such loop
     */
    public static LoopSignature find(String op, DType... inputs) {
        Table t = TABLES.get(op);
        return t == null ? null : t.signatures.get(key(List.of(inputs)));
    }

    /**
     * Returns the one-input loop for exactly this input dtype.
     *
     * @param op operation name
     * @param in input dtype
     * @return the loop
     * @throws DTypeException if there is none
     */
    public static UnaryLoop unary(String op, DType in) {
        Object loop = table(op).loops.get(key(in));
        if (!(loop instanceof UnaryLoop u)) {
            throw noLoop(op, in);
        }
        return u;
    }

    /**
     * Returns the two-input loop for exactly these input dtypes.
     *
     * @param op  operation name
     * @param in1 first input dtype
     * @param in2 second input dtype
     * @return the loop
     * @throws DTypeException if there is none
     */
    public static BinaryLoop binary(String op, DType in1, DType in2) {
        Object loop = table(op).loops.get(key(in1, in2));
        if (!(loop instanceof BinaryLoop b)) {
            throw noLoop(op, in1, in2);
        }
        return b;
    }

    private static DTypeException noLoop(String op, DType... inputs) {
        String types = java.util.Arrays.stream(inputs).map(DType::typeName).collect(Collectors.joining(", "));
        return new DTypeException("operation '" + op + "' has no loop for (" + types + "); available: "
                + table(op).order.stream().map(s -> s.inputs().stream().map(DType::typeName)
                .collect(Collectors.joining(", ", "(", ")"))).collect(Collectors.joining(" ")));
    }
}
