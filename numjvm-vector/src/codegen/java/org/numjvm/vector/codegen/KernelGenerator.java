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
            String layout = "java.lang.foreign.ValueLayout.JAVA_" + switch (dtype) {
                case "FLOAT32" -> "FLOAT"; case "FLOAT64" -> "DOUBLE"; case "INT32" -> "INT"; default -> "LONG";
            } + "_UNALIGNED";
            int bytes = dtype.endsWith("32") ? 4 : 8;
            String name = dtype.toLowerCase() + "_" + operation;
            String template = Files.readString(root.resolve(java.util.Set.of("equal", "not_equal", "less", "less_equal", "greater", "greater_equal").contains(operation) ? "Comparison.java.tmpl" : "Binary.java.tmpl"));
            methods.append(template.replace("${NAME}", name).replace("${VECTOR}", vector)
                    .replace("${LAYOUT}", layout).replace("${BYTES}", "" + bytes).replace("${EXPRESSION}", expression));
            cases.add("case \"" + dtype + ":" + operation + "\" -> " + name + "(scalar);\n");
        }
        var unaryCases = new ArrayList<String>();
        for (String dtype : new String[]{"FLOAT32", "FLOAT64"}) {
            for (String op : new String[]{"negative", "absolute", "square", "sqrt"}) {
                String expression = switch(op) {
                    case "negative" -> "x.neg()"; case "absolute" -> "x.abs()";
                    case "square" -> "x.mul(x)"; default -> "x.lanewise(VectorOperators.SQRT)";
                };
                String name = dtype.toLowerCase() + "_" + op;
                methods.append(Files.readString(root.resolve("Unary.java.tmpl"))
                    .replace("${NAME}", name).replace("${VECTOR}", dtype.equals("FLOAT32") ? "FloatVector" : "DoubleVector")
                    .replace("${BYTES}", dtype.equals("FLOAT32") ? "4" : "8").replace("${EXPRESSION}", expression));
                unaryCases.add("case \"" + dtype + ":" + op + "\" -> " + name + "(scalar);\n");
            }
        }
        String source = "package org.numjvm.vector;\nimport java.nio.ByteOrder;\nimport jdk.incubator.vector.*;\n"
                + "import org.numjvm.dtype.DType;\nimport org.numjvm.internal.backend.Accelerators;\n"
                + "import org.numjvm.internal.loops.BinaryLoop;\nimport org.numjvm.internal.loops.UnaryLoop;\nfinal class GeneratedVectorKernels {\n"
                + "private GeneratedVectorKernels() {}\nstatic BinaryLoop binary(String op,DType type,BinaryLoop scalar) {\n"
                + "return switch(type.name()+\":\"+op) {\n" + String.join("", cases)
                + "default -> scalar; }; }\n"
                + "static UnaryLoop unary(String op,DType type,UnaryLoop scalar) { return switch(type.name()+\":\"+op) {\n"
                + String.join("", unaryCases) + "default -> scalar; }; }\n" + methods + "}\n";
        Path out = Path.of(args[1]).resolve("org/numjvm/vector/GeneratedVectorKernels.java");
        Files.createDirectories(out.getParent()); Files.writeString(out, source);
    }
}
