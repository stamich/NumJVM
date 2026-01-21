package org.numjvm.vector.codegen;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/** Expands auditable per-operation templates; generated sources stay in build/. */
public final class KernelGenerator {
    private KernelGenerator() { }
    public static void main(String[] args) throws Exception {
        Path root = Path.of(args[0]);
        var methods = new StringBuilder();
        var cases = new ArrayList<String>();
        for (String line : Files.readAllLines(root.resolve("binary.tsv"))) {
            if (line.isBlank() || line.startsWith("#")) { continue; }
            String[] parts = line.split("\\s+", 3);
            String dtype = parts[0], operation = parts[1], expression = parts[2];
            String vector = switch (dtype) {
                case "FLOAT32" -> "FloatVector"; case "FLOAT64" -> "DoubleVector";
                case "INT32" -> "IntVector"; case "INT64" -> "LongVector";
                default -> throw new IllegalArgumentException(dtype);
            };
            int bytes = dtype.endsWith("32") ? 4 : 8;
            String name = dtype.toLowerCase() + "_" + operation;
            String template = Files.readString(root.resolve("Binary.java.tmpl"));
            methods.append(template.replace("${NAME}", name).replace("${VECTOR}", vector)
                    .replace("${BYTES}", "" + bytes).replace("${EXPRESSION}", expression));
            cases.add("case \"" + dtype + ":" + operation + "\" -> " + name + "(scalar);\n");
        }
        String source = "package org.numjvm.vector;\nimport java.nio.ByteOrder;\nimport jdk.incubator.vector.*;\n"
                + "import org.numjvm.dtype.DType;\nimport org.numjvm.internal.backend.Accelerators;\n"
                + "import org.numjvm.internal.loops.BinaryLoop;\nfinal class GeneratedVectorKernels {\n"
                + "private GeneratedVectorKernels() {}\nstatic BinaryLoop binary(String op,DType type,BinaryLoop scalar) {\n"
                + "return switch(type.name()+\":\"+op) {\n" + String.join("", cases)
                + "default -> scalar; }; }\n" + methods + "}\n";
        Path out = Path.of(args[1]).resolve("org/numjvm/vector/GeneratedVectorKernels.java");
        Files.createDirectories(out.getParent()); Files.writeString(out, source);
    }
}
