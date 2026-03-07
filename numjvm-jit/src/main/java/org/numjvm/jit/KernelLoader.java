package org.numjvm.jit;

import java.lang.classfile.ClassFile;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;

/** Builds and defines a hidden class in the same module as its kernel interface. */
final class KernelLoader {
    private static final int MAX_CLASS_BYTES = 65536;
    private KernelLoader() { }

    static GeneratedKernel create() {
        ClassDesc name = ClassDesc.of("org.numjvm.jit.GeneratedExpression");
        ClassDesc marker = ClassDesc.of("org.numjvm.jit.GeneratedKernel");
        byte[] bytes = ClassFile.of().build(name, builder -> builder
                .withFlags(ClassFile.ACC_FINAL | ClassFile.ACC_SUPER)
                .withInterfaceSymbols(marker)
                .withMethodBody("<init>", MethodTypeDesc.of(ConstantDescs.CD_void), 0,
                        code -> code.aload(0).invokespecial(ConstantDescs.CD_Object, "<init>",
                                MethodTypeDesc.of(ConstantDescs.CD_void)).return_())
                .withMethodBody("version", MethodTypeDesc.of(ConstantDescs.CD_int), ClassFile.ACC_PUBLIC,
                        code -> code.iconst_1().ireturn()));
        if (bytes.length > MAX_CLASS_BYTES) throw new IllegalArgumentException("generated class too large");
        try {
            var lookup = MethodHandles.lookup().defineHiddenClass(bytes, true);
            return (GeneratedKernel) lookup.findConstructor(lookup.lookupClass(),
                    java.lang.invoke.MethodType.methodType(void.class)).invoke();
        } catch (RuntimeException | Error e) {
            throw e;
        } catch (Throwable e) {
            throw new IllegalStateException("failed to define generated kernel", e);
        }
    }
}
