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

/** Generates a typed vector expression body with Java's Class-File API. */
final class FloatExpressionCompiler {
    private static final ClassDesc VECTOR = ClassDesc.of("jdk.incubator.vector.FloatVector");
    private static final ClassDesc BASE_VECTOR = ClassDesc.of("jdk.incubator.vector.Vector");
    private static final ClassDesc SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private FloatExpressionCompiler() { }

    static FloatExpression compile(FusedGraph graph) {
        if (graph.nodes().size() > 64 || graph.inputTypes().stream().anyMatch(t -> t != DType.FLOAT32)) return null;
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i);
            if ((node.type() != DType.FLOAT32 && node.op() != FusedGraph.Op.CONSTANT) || node.left() >= i || node.right() >= i
                    || node.left() < -1 || node.right() < -1) return null;
            if (node.op() == FusedGraph.Op.INPUT && (node.slot() < 0 || node.slot() >= graph.inputTypes().size())) return null;
            boolean binary = switch (node.op()) { case ADD, SUBTRACT, MULTIPLY, DIVIDE -> true; default -> false; };
            boolean unary = switch (node.op()) { case NEGATIVE, ABSOLUTE, SQUARE, SQRT -> true; default -> false; };
            if ((binary && (node.left() < 0 || node.right() < 0)) || (unary && node.left() < 0)) return null;
        }
        ClassDesc name = ClassDesc.of("org.numjvm.vector.GeneratedFloatExpression");
        ClassDesc contract = ClassDesc.of("org.numjvm.vector.FloatExpression");
        byte[] bytes = ClassFile.of().build(name, builder -> builder.withFlags(ClassFile.ACC_FINAL | ClassFile.ACC_SUPER)
                .withInterfaceSymbols(contract)
                .withMethodBody("<init>", MethodTypeDesc.of(ConstantDescs.CD_void), 0,
                        code -> code.aload(0).invokespecial(ConstantDescs.CD_Object, "<init>", MethodTypeDesc.of(ConstantDescs.CD_void)).return_())
                .withMethodBody("evaluate", MethodTypeDesc.of(VECTOR, VECTOR.arrayType()), ClassFile.ACC_PUBLIC,
                        code -> emit(code, graph)));
        try {
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
            return (FloatExpression) lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class)).invoke();
        } catch (RuntimeException | Error error) { throw error; }
        catch (Throwable error) { throw new IllegalStateException("cannot define fused vector expression", error); }
    }

    private static void emit(CodeBuilder code, FusedGraph graph) {
        for (int i = 0; i < graph.nodes().size(); i++) {
            var node = graph.nodes().get(i);
            switch (node.op()) {
                case INPUT -> code.aload(1).loadConstant(node.slot()).aaload();
                case CONSTANT -> code.getstatic(VECTOR, "SPECIES_PREFERRED", SPECIES)
                        .loadConstant(Float.floatToRawIntBits((float) Double.longBitsToDouble(node.bits())))
                        .invokestatic(ClassDesc.of("java.lang.Float"), "intBitsToFloat",
                                MethodTypeDesc.of(ConstantDescs.CD_float, ConstantDescs.CD_int))
                        .invokestatic(VECTOR, "broadcast", MethodTypeDesc.of(VECTOR, SPECIES, ConstantDescs.CD_float));
                case ADD, SUBTRACT, MULTIPLY, DIVIDE -> {
                    String method = switch (node.op()) { case ADD -> "add"; case SUBTRACT -> "sub";
                        case MULTIPLY -> "mul"; default -> "div"; };
                    code.aload(2 + node.left()).aload(2 + node.right())
                            .invokevirtual(VECTOR, method, MethodTypeDesc.of(VECTOR, BASE_VECTOR));
                }
                case NEGATIVE, ABSOLUTE, SQRT -> {
                    String method = node.op() == FusedGraph.Op.NEGATIVE ? "neg" : node.op() == FusedGraph.Op.ABSOLUTE ? "abs" : "sqrt";
                    code.aload(2 + node.left()).invokevirtual(VECTOR, method, MethodTypeDesc.of(VECTOR));
                }
                case SQUARE -> code.aload(2 + node.left()).aload(2 + node.left())
                        .invokevirtual(VECTOR, "mul", MethodTypeDesc.of(VECTOR, BASE_VECTOR));
            }
            code.astore(2 + i);
        }
        code.aload(1 + graph.nodes().size()).areturn();
    }
}
