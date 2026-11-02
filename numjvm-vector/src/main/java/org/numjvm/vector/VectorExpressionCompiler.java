package org.numjvm.vector;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import org.numjvm.dtype.DType;
import org.numjvm.internal.backend.FusedGraph;

/** Generates homogeneous typed vector expressions with Java's Class-File API. */
final class VectorExpressionCompiler {
    private static final ClassDesc BASE_VECTOR = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private VectorExpressionCompiler() { }

    static Object compile(FusedGraph graph, DType type) {
        boolean single = type == DType.FLOAT32;
        if ((!single && type != DType.FLOAT64) || graph.nodes().size() > 64
                || graph.inputTypes().stream().anyMatch(t -> t != type)) return null;
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i);
            if ((node.type() != type && !(single && node.op() == FusedGraph.Op.CONSTANT))
                    || node.left() >= i || node.right() >= i || node.left() < -1 || node.right() < -1) return null;
            if (node.op() == FusedGraph.Op.INPUT && (node.slot() < 0 || node.slot() >= graph.inputTypes().size())) return null;
            boolean binary = switch (node.op()) { case ADD, SUBTRACT, MULTIPLY, DIVIDE -> true; default -> false; };
            boolean unary = switch (node.op()) { case NEGATIVE, ABSOLUTE, SQUARE, SQRT -> true; default -> false; };
            if ((binary && (node.left() < 0 || node.right() < 0)) || (unary && node.left() < 0)) return null;
        }
        String kind = single ? "Float" : "Double";
        ClassDesc vector = ClassDesc.of("jdk.incubator.vector." + kind + "Vector");
        ClassDesc name = ClassDesc.of("org.numjvm.vector.Generated" + kind + "Expression");
        ClassDesc contract = ClassDesc.of("org.numjvm.vector." + kind + "Expression");
        byte[] bytes = ClassFile.of().build(name, builder -> builder.withFlags(ClassFile.ACC_FINAL | ClassFile.ACC_SUPER)
                .withInterfaceSymbols(contract, ClassDesc.of("org.numjvm.vector.VectorLoop"))
                .withMethodBody("<init>", MethodTypeDesc.of(ConstantDescs.CD_void), 0,
                        code -> code.aload(0).invokespecial(ConstantDescs.CD_Object, "<init>", MethodTypeDesc.of(ConstantDescs.CD_void)).return_())
                .withMethodBody("evaluate", MethodTypeDesc.of(vector, vector.arrayType()), ClassFile.ACC_PUBLIC,
                        code -> { emit(code, graph, vector, single, 2, null, null, null); code.areturn(); })
                .withMethodBody("runVectors", MethodTypeDesc.of(ConstantDescs.CD_void,
                                ClassDesc.of("java.lang.foreign.MemorySegment").arrayType(),
                                ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long.arrayType(),
                                ConstantDescs.CD_long, ConstantDescs.CD_int), ClassFile.ACC_PUBLIC,
                        code -> emitLoop(code, graph, vector, single)));
        try {
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
            return lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class)).invoke();
        } catch (RuntimeException | Error error) { throw error; }
        catch (Throwable error) { throw new IllegalStateException("cannot define fused vector expression", error); }
    }

    private static void emitLoop(CodeBuilder code, FusedGraph graph, ClassDesc vector, boolean single) {
        int count = graph.inputTypes().size() + 1;
        int[] segments = new int[count], offsets = new int[count], strides = new int[count];
        int local = 9;
        for (int i = 0; i < count; i++) {
            segments[i] = local++; offsets[i] = local; local += 2; strides[i] = local; local += 2;
            code.aload(1).loadConstant(i).aaload().astore(segments[i]);
            code.aload(2).loadConstant(i).laload().lstore(offsets[i]);
            code.aload(3).loadConstant(i).laload().lstore(strides[i]);
        }
        var loop = code.newLabel();
        var done = code.newLabel();
        code.lconst_0().lstore(7).labelBinding(loop).lload(7).lload(4).lcmp().ifge(done);
        code.aload(segments[count - 1]).lload(offsets[count - 1]);
        emit(code, graph, vector, single, local, segments, offsets, strides);
        code.invokestatic(ClassDesc.of("org.numjvm.vector.VectorMemory"), single ? "storeFloat" : "storeDouble",
                MethodTypeDesc.of(ConstantDescs.CD_void, ClassDesc.of("java.lang.foreign.MemorySegment"), ConstantDescs.CD_long, vector));
        for (int i = 0; i < count; i++)
            code.lload(offsets[i]).lload(strides[i]).iload(6).i2l().lmul().ladd().lstore(offsets[i]);
        code.lload(7).iload(6).i2l().ladd().lstore(7).goto_(loop).labelBinding(done).return_();
    }

    private static void emit(CodeBuilder code, FusedGraph graph, ClassDesc vector, boolean single,
                             int first, int[] segments, int[] offsets, int[] strides) {
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i);
            switch (node.op()) {
                case INPUT -> {
                    if (segments == null) code.aload(1).loadConstant(node.slot()).aaload();
                    else code.aload(segments[node.slot()]).lload(offsets[node.slot()]).lload(strides[node.slot()])
                            .invokestatic(ClassDesc.of("org.numjvm.vector.VectorMemory"), single ? "loadFloat" : "loadDouble",
                                    MethodTypeDesc.of(vector, ClassDesc.of("java.lang.foreign.MemorySegment"),
                                            ConstantDescs.CD_long, ConstantDescs.CD_long));
                }
                case CONSTANT -> {
                    code.getstatic(vector, "SPECIES_PREFERRED", SPECIES);
                    if (single) {
                        code.loadConstant(Float.floatToRawIntBits((float) Double.longBitsToDouble(node.bits())))
                                .invokestatic(ClassDesc.of("java.lang.Float"), "intBitsToFloat",
                                        MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                                .invokestatic(vector, "broadcast", MethodTypeDesc.of(vector, SPECIES, ConstantDescs.CD_float));
                    } else {
                        code.loadConstant(node.bits()).invokestatic(ClassDesc.of("java.lang.Double"), "longBitsToDouble",
                                        MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_long))
                                .invokestatic(vector, "broadcast", MethodTypeDesc.of(vector, SPECIES, ConstantDescs.CD_double));
                    }
                }
                case ADD, SUBTRACT, MULTIPLY, DIVIDE -> {
                    String method = switch (node.op()) { case ADD -> "add"; case SUBTRACT -> "sub";
                        case MULTIPLY -> "mul"; default -> "div"; };
                    code.aload(first + node.left()).aload(first + node.right())
                            .invokevirtual(vector, method, MethodTypeDesc.of(vector, BASE_VECTOR));
                }
                case NEGATIVE, ABSOLUTE, SQRT -> {
                    String method = node.op() == FusedGraph.Op.NEGATIVE ? "neg" : node.op() == FusedGraph.Op.ABSOLUTE ? "abs" : "sqrt";
                    code.aload(first + node.left()).invokevirtual(vector, method, MethodTypeDesc.of(vector));
                }
                case SQUARE -> code.aload(first + node.left()).aload(first + node.left())
                        .invokevirtual(vector, "mul", MethodTypeDesc.of(vector, BASE_VECTOR));
            }
            code.astore(first + i);
        }
        code.aload(first + graph.nodes().size() - 1);
    }
}
