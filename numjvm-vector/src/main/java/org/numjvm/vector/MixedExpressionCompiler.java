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

/** Emits vector operations and explicit per-node promotion for mixed graphs. */
final class MixedExpressionCompiler {
    private static final ClassDesc FLOAT = ClassDesc.of("jdk.incubator.vector.FloatVector");
    private static final ClassDesc DOUBLE = ClassDesc.of("jdk.incubator.vector.DoubleVector");
    private static final ClassDesc VECTOR = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private static final ClassDesc CONVERT = ClassDesc.of("org.numjvm.vector.MixedConversions");
    private MixedExpressionCompiler() { }

    static MixedExpression compile(FusedGraph graph) {
        if (MixedConversions.FLOAT == null || graph.nodes().size() > 64
                || graph.nodes().getLast().type() != DType.FLOAT64) return null;
        for (int i = 0; i < graph.nodes().size(); i++) {
            var n = graph.nodes().get(i);
            if ((n.type() != DType.FLOAT32 && n.type() != DType.FLOAT64)
                    || n.left() >= i || n.right() >= i || n.left() < -1 || n.right() < -1) return null;
            if (n.op() == FusedGraph.Op.INPUT && (n.slot() < 0 || n.slot() >= graph.inputTypes().size()
                    || n.type() != graph.inputTypes().get(n.slot()))) return null;
            boolean binary = switch (n.op()) { case ADD, SUBTRACT, MULTIPLY, DIVIDE -> true; default -> false; };
            boolean unary = switch (n.op()) { case NEGATIVE, ABSOLUTE, SQUARE, SQRT -> true; default -> false; };
            if ((binary && (n.left() < 0 || n.right() < 0)) || (unary && n.left() < 0)) return null;
        }
        ClassDesc name = ClassDesc.of("org.numjvm.vector.GeneratedMixedExpression");
        byte[] bytes = ClassFile.of().build(name, builder -> builder.withFlags(ClassFile.ACC_FINAL | ClassFile.ACC_SUPER)
                .withInterfaceSymbols(ClassDesc.of("org.numjvm.vector.MixedExpression"))
                .withMethodBody("<init>", MethodTypeDesc.of(ConstantDescs.CD_void), 0,
                        code -> code.aload(0).invokespecial(ConstantDescs.CD_Object, "<init>", MethodTypeDesc.of(ConstantDescs.CD_void)).return_())
                .withMethodBody("evaluate", MethodTypeDesc.of(DOUBLE, ConstantDescs.CD_Object.arrayType()), ClassFile.ACC_PUBLIC,
                        code -> emit(code, graph)));
        try {
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
            return (MixedExpression) lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class)).invoke();
        } catch (RuntimeException | Error error) { throw error; }
        catch (Throwable error) { throw new IllegalStateException("cannot define mixed fused expression", error); }
    }

    private static void operand(CodeBuilder code, FusedGraph graph, int index, DType target) {
        DType source = graph.nodes().get(index).type();
        code.aload(2 + index);
        if (source == DType.FLOAT32 && target == DType.FLOAT64)
            code.invokestatic(CONVERT, "toDouble", MethodTypeDesc.of(DOUBLE, FLOAT));
        else if (source == DType.FLOAT64 && target == DType.FLOAT32)
            code.invokestatic(CONVERT, "toFloat", MethodTypeDesc.of(FLOAT, DOUBLE));
    }

    private static void emit(CodeBuilder code, FusedGraph graph) {
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i);
            ClassDesc type = node.type() == DType.FLOAT32 ? FLOAT : DOUBLE;
            switch (node.op()) {
                case INPUT -> code.aload(1).loadConstant(node.slot()).aaload().checkcast(type);
                case CONSTANT -> {
                    code.getstatic(CONVERT, "DOUBLE", SPECIES).loadConstant(node.bits())
                            .invokestatic(ClassDesc.of("java.lang.Double"), "longBitsToDouble",
                                    MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_long))
                            .invokestatic(DOUBLE, "broadcast", MethodTypeDesc.of(DOUBLE, SPECIES, ConstantDescs.CD_double));
                    if (node.type() == DType.FLOAT32) code.invokestatic(CONVERT, "toFloat", MethodTypeDesc.of(FLOAT, DOUBLE));
                }
                case ADD, SUBTRACT, MULTIPLY, DIVIDE -> {
                    String method = switch (node.op()) { case ADD -> "add"; case SUBTRACT -> "sub";
                        case MULTIPLY -> "mul"; default -> "div"; };
                    operand(code, graph, node.left(), node.type()); operand(code, graph, node.right(), node.type());
                    code.invokevirtual(type, method, MethodTypeDesc.of(type, VECTOR));
                }
                case NEGATIVE, ABSOLUTE, SQRT -> {
                    String method = node.op() == FusedGraph.Op.NEGATIVE ? "neg" : node.op() == FusedGraph.Op.ABSOLUTE ? "abs" : "sqrt";
                    operand(code, graph, node.left(), node.type());
                    code.invokevirtual(type, method, MethodTypeDesc.of(type));
                }
                case SQUARE -> {
                    operand(code, graph, node.left(), node.type()); operand(code, graph, node.left(), node.type());
                    code.invokevirtual(type, "mul", MethodTypeDesc.of(type, VECTOR));
                }
            }
            code.astore(2 + i);
        }
        code.aload(1 + graph.nodes().size()).areturn();
    }
}
