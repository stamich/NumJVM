package org.numjvm.jit;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.IdentityHashMap;
import java.util.List;
import org.numjvm.dtype.DType;

/** Emits a hidden expression class in the same module as its kernel interface. */
final class KernelLoader {
    private static final int MAX_CLASS_BYTES = 65536;
    private static final int MAX_GENERATED_NODES = 64;
    private KernelLoader() { }

    static GeneratedKernel create(Expression expression, DType[] inputTypes) {
        var types = new TypeResolver(inputTypes);
        List<Expr> nodes = new GraphPlan(expression.root()).nodes();
        if (nodes.size() > MAX_GENERATED_NODES) throw new IllegalArgumentException("expression exceeds code generation limit");
        ClassDesc name = ClassDesc.of("org.numjvm.jit.GeneratedExpression");
        ClassDesc marker = ClassDesc.of("org.numjvm.jit.GeneratedKernel");
        byte[] bytes = ClassFile.of().build(name, builder -> builder
                .withFlags(ClassFile.ACC_FINAL | ClassFile.ACC_SUPER)
                .withInterfaceSymbols(marker)
                .withMethodBody("<init>", MethodTypeDesc.of(ConstantDescs.CD_void), 0,
                        code -> code.aload(0).invokespecial(ConstantDescs.CD_Object, "<init>",
                                MethodTypeDesc.of(ConstantDescs.CD_void)).return_())
                .withMethodBody("version", MethodTypeDesc.of(ConstantDescs.CD_int), ClassFile.ACC_PUBLIC,
                        code -> code.iconst_1().ireturn())
                .withMethodBody("evaluate", MethodTypeDesc.of(ConstantDescs.CD_double,
                                ConstantDescs.CD_double.arrayType()), ClassFile.ACC_PUBLIC,
                        code -> emit(code, nodes, types))
                .withMethodBody("run", MethodTypeDesc.of(ConstantDescs.CD_void,
                                ClassDesc.of("java.lang.foreign.MemorySegment").arrayType(),
                                ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long.arrayType(), ConstantDescs.CD_long),
                        ClassFile.ACC_PUBLIC, code -> emitRun(code, marker, inputTypes, types.resolve(expression.root()))));
        if (bytes.length > MAX_CLASS_BYTES) throw new IllegalArgumentException("generated class too large");
        try {
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
            return (GeneratedKernel) lookup.findConstructor(lookup.lookupClass(), MethodType.methodType(void.class)).invoke();
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable e) {
            throw new IllegalStateException("failed to define generated kernel", e);
        }
    }

    static boolean supports(Expression expression) {
        return new GraphPlan(expression.root()).nodes().size() <= MAX_GENERATED_NODES;
    }

    private static void emitRun(CodeBuilder code, ClassDesc kernel, DType[] inputs, DType result) {
        ClassDesc memory = ClassDesc.of("java.lang.foreign.MemorySegment");
        ClassDesc access = ClassDesc.of("org.numjvm.jit.GeneratedAccess");
        var read = MethodTypeDesc.of(ConstantDescs.CD_double, memory, ConstantDescs.CD_long);
        var write = MethodTypeDesc.of(ConstantDescs.CD_void, memory, ConstantDescs.CD_long, ConstantDescs.CD_double);
        var loop = code.newLabel();
        var done = code.newLabel();
        code.loadConstant(inputs.length).newarray(java.lang.classfile.TypeKind.DOUBLE).astore(8);
        code.lconst_0().lstore(6).labelBinding(loop).lload(6).lload(4).lcmp().ifge(done);
        for (int i = 0; i < inputs.length; i++) {
            code.aload(8).loadConstant(i).aload(1).loadConstant(i).aaload();
            offset(code, i);
            code.invokestatic(access, inputs[i] == DType.FLOAT32 ? "readFloat" : "readDouble", read).dastore();
        }
        code.aload(1).loadConstant(inputs.length).aaload();
        offset(code, inputs.length);
        code.aload(0).aload(8).invokeinterface(kernel, "evaluate",
                MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_double.arrayType()));
        code.invokestatic(access, result == DType.FLOAT32 ? "writeFloat" : "writeDouble", write);
        code.lload(6).lconst_1().ladd().lstore(6).goto_(loop).labelBinding(done).return_();
    }

    private static void offset(CodeBuilder code, int slot) {
        code.aload(2).loadConstant(slot).laload().lload(6).aload(3).loadConstant(slot).laload().lmul().ladd();
    }

    private static void emit(CodeBuilder code, List<Expr> nodes, TypeResolver types) {
        var locals = new IdentityHashMap<Expr, Integer>();
        int next = 2;
        for (Expr node : nodes) {
            DType type = types.resolve(node);
            boolean single = type == DType.FLOAT32;
            switch (node.op) {
                case INPUT -> code.aload(1).loadConstant(node.slot).daload();
                case CONSTANT -> code.loadConstant(node.bits).invokestatic(ClassDesc.of("java.lang.Double"),
                        "longBitsToDouble", MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_long));
                case ADD, SUBTRACT, MULTIPLY, DIVIDE -> {
                    code.dload(locals.get(node.left));
                    if (single) code.d2f();
                    code.dload(locals.get(node.right));
                    if (single) code.d2f();
                    switch (node.op) {
                        case ADD -> { if (single) code.fadd(); else code.dadd(); }
                        case SUBTRACT -> { if (single) code.fsub(); else code.dsub(); }
                        case MULTIPLY -> { if (single) code.fmul(); else code.dmul(); }
                        case DIVIDE -> { if (single) code.fdiv(); else code.ddiv(); }
                        default -> throw new AssertionError();
                    }
                    if (single) code.f2d();
                }
                case NEGATIVE -> code.dload(locals.get(node.left)).dneg();
                case ABSOLUTE -> code.dload(locals.get(node.left)).invokestatic(ClassDesc.of("java.lang.Math"),
                        "abs", MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_double));
                case SQRT -> code.dload(locals.get(node.left)).invokestatic(ClassDesc.of("java.lang.Math"),
                        "sqrt", MethodTypeDesc.of(ConstantDescs.CD_double, ConstantDescs.CD_double));
                case SQUARE -> {
                    code.dload(locals.get(node.left));
                    if (single) code.d2f();
                    code.dload(locals.get(node.left));
                    if (single) code.d2f();
                    if (single) code.fmul().f2d(); else code.dmul();
                }
            }
            if (single && (node.op == Expr.Op.INPUT || node.op == Expr.Op.SQRT)) code.d2f().f2d();
            code.dstore(next);
            locals.put(node, next);
            next += 2;
        }
        code.dload(locals.get(nodes.getLast())).dreturn();
    }
}
