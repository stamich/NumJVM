package org.numjvm.codegen;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Generates NumJVM's scalar inner loops (decision D3) from the templates in
 * {@code numjvm-core/src/codegen/templates} and the operation table in {@link Ops}.
 *
 * <p>For every real dtype it writes one class {@code org.numjvm.internal.loops.<Type>Loops}
 * holding that dtype's loops, plus {@code GeneratedLoops}, which registers them all in
 * the kernel registry. Run by the {@code generateKernels} Gradle task before
 * {@code compileJava}:
 *
 * <pre>
 * java org.numjvm.codegen.KernelGenerator &lt;templates dir&gt; &lt;output source root&gt;
 * </pre>
 */
public final class KernelGenerator {

    private static final Pattern PLACEHOLDER = Pattern.compile("\\$\\{([A-Z_]+)}");
    private static final String PACKAGE_DIR = "org/numjvm/internal/loops";

    private final Map<String, String> templates = new HashMap<>();

    private KernelGenerator(Path templateDir) throws IOException {
        for (String name : List.of("LoopClass", "BinaryLoop", "UnaryLoop", "CopyLoop", "GeneratedLoops", "ReducePath",
                "PairwiseReducePath")) {
            templates.put(name, Files.readString(templateDir.resolve(name + ".java.tmpl"), StandardCharsets.UTF_8));
        }
    }

    /**
     * Entry point.
     *
     * @param args template directory and output source root
     * @throws IOException if reading templates or writing sources fails
     */
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("usage: KernelGenerator <templates dir> <output source root>");
            System.exit(2);
        }
        Path out = Path.of(args[1]).resolve(PACKAGE_DIR);
        Files.createDirectories(out);
        try (var old = Files.list(out)) {
            for (Path p : old.toList()) {
                Files.delete(p);
            }
        }
        int loops = new KernelGenerator(Path.of(args[0])).generate(out);
        System.out.println("generated " + loops + " loops in " + out);
    }

    private int generate(Path out) throws IOException {
        StringBuilder registerAll = new StringBuilder();
        int count = 0;
        for (DTypeSpec t : DTypeSpec.REAL) {
            List<String> registrations = new ArrayList<>();
            StringBuilder methods = new StringBuilder();
            for (OpSpec op : Ops.ALL) {
                String expr = op.expression(t);
                if (expr == null) {
                    continue;
                }
                methods.append(loop(op, t, expr));
                DTypeSpec o = op.output(t);
                registrations.add(op.arity() == 2
                        ? "        r.binary(\"%s\", DType.%s, DType.%s, DType.%s, %sLoops::%s);"
                        .formatted(op.name(), t.dtype(), t.dtype(), o.dtype(), t.cls(), method(op))
                        : "        r.unary(\"%s\", DType.%s, DType.%s, %sLoops::%s);"
                        .formatted(op.name(), t.dtype(), o.dtype(), t.cls(), method(op)));
                count++;
            }
            for (DTypeSpec u : DTypeSpec.REAL) {
                if (u == t) {
                    registrations.add("        r.unary(\"cast:%s\", DType.%s, DType.%s, %sLoops::copy);"
                            .formatted(u.dtype().toLowerCase(), t.dtype(), u.dtype(), t.cls()));
                } else {
                    methods.append(cast(t, u.dtype(), u.size(), "to" + u.cls(),
                            (seg, off) -> Casts.write(t, u, seg, off, "x")));
                    registrations.add("        r.unary(\"cast:%s\", DType.%s, DType.%s, %sLoops::to%s);"
                            .formatted(u.dtype().toLowerCase(), t.dtype(), u.dtype(), t.cls(), u.cls()));
                }
                count++;
            }
            for (boolean single : new boolean[]{true, false}) {
                String c = single ? "COMPLEX64" : "COMPLEX128";
                String cls = single ? "Complex64" : "Complex128";
                methods.append(cast(t, c, single ? 8 : 16, "to" + cls,
                        (seg, off) -> Casts.writeComplex(t, single, seg, off, "x")));
                registrations.add("        r.unary(\"cast:%s\", DType.%s, DType.%s, %sLoops::to%s);"
                        .formatted(c.toLowerCase(), t.dtype(), c, t.cls(), cls));
                count++;
            }
            methods.append(fill("CopyLoop", Map.of(
                    "SIGNATURE", "copy(" + t.dtype().toLowerCase() + ") -> " + t.dtype().toLowerCase(),
                    "ISIZE", String.valueOf(t.size()),
                    "BYTES", t.size() == 1 ? "n" : "n * " + t.size(),
                    "LAYOUT", t.layout())));
            registrations.add("        r.unary(\"copy\", DType.%s, DType.%s, %sLoops::copy);"
                    .formatted(t.dtype(), t.dtype(), t.cls()));
            count++;
            String body = methods.toString();
            String imports = (body.contains("Unsigned.") ? "import org.numjvm.dtype.Unsigned;\n" : "")
                    + (body.contains("HalfFloat.") ? "import org.numjvm.internal.HalfFloat;\n" : "");
            String src = fill("LoopClass", Map.of(
                    "IMPORTS", imports,
                    "TYPENAME", t.dtype().toLowerCase(),
                    "CLASS", t.cls() + "Loops",
                    "REGISTRATIONS", String.join("\n", registrations),
                    "METHODS", body));
            Files.writeString(out.resolve(t.cls() + "Loops.java"), src, StandardCharsets.UTF_8);
            registerAll.append("        ").append(t.cls()).append("Loops.register(r);\n");
        }
        Files.writeString(out.resolve("GeneratedLoops.java"), fill("GeneratedLoops", Map.of(
                "COUNT", String.valueOf(count),
                "REGISTRATIONS", registerAll.toString().stripTrailing())), StandardCharsets.UTF_8);
        return count;
    }

    private static String method(OpSpec op) {
        StringBuilder sb = new StringBuilder();
        boolean upper = false;
        for (char c : op.name().toCharArray()) {
            if (c == '_') {
                upper = true;
            } else {
                sb.append(upper ? Character.toUpperCase(c) : c);
                upper = false;
            }
        }
        return sb.toString();
    }

    private String loop(OpSpec op, DTypeSpec t, String expr) {
        DTypeSpec o = op.output(t);
        String in = t.dtype().toLowerCase();
        String signature = op.name() + "(" + (op.arity() == 2 ? in + ", " + in : in) + ") -> "
                + o.dtype().toLowerCase();
        Map<String, String> v = new HashMap<>();
        v.put("OP", op.name());
        v.put("DOC", html(op.doc()));
        v.put("SIGNATURE", signature);
        v.put("METHOD", method(op));
        v.put("CT", t.ct());
        v.put("ISIZE", String.valueOf(t.size()));
        v.put("OSIZE", String.valueOf(o.size()));
        v.put("READ_A", t.read("a", "pa"));
        v.put("READ_A_I", t.read("a", indexed("pa", t.size())));
        v.put("READ_B", t.read("b", "pb"));
        v.put("READ_B_I", t.read("b", indexed("pb", t.size())));
        v.put("WRITE", o.write("o", "po", expr));
        v.put("WRITE_I", o.write("o", indexed("po", o.size()), expr));
        // binary operations whose output dtype is the input dtype can reduce (ADR-004): the
        // executor passes the output as the first input with stride 0
        boolean reduces = op.arity() == 2 && !op.boolOutput();
        v.put("EXPR", t.narrow(expr));
        v.put("WRITE_X", o.write("o", "po", "x"));
        // add reduces floating-point runs with NumPy's pairwise summation
        boolean pairwise = reduces && op.name().equals("add") && t.isFloat();
        v.put("PAIRWISE", pairwise ? "Pairwise." + t.dtype().toLowerCase() : "");
        v.put("REDUCE", pairwise ? fill("PairwiseReducePath", v) : reduces ? fill("ReducePath", v) : "");
        return fill(op.arity() == 2 ? "BinaryLoop" : "UnaryLoop", v);
    }

    private String cast(DTypeSpec t, String to, int outSize, String method,
                        java.util.function.BinaryOperator<String> write) {
        Map<String, String> v = new HashMap<>();
        v.put("OP", "cast");
        v.put("DOC", "conversion to " + to.toLowerCase() + " with the rules of astype");
        v.put("SIGNATURE", "cast(" + t.dtype().toLowerCase() + ") -> " + to.toLowerCase());
        v.put("METHOD", method);
        v.put("CT", t.ct());
        v.put("ISIZE", String.valueOf(t.size()));
        v.put("OSIZE", String.valueOf(outSize));
        v.put("READ_A", t.read("a", "pa"));
        v.put("READ_A_I", t.read("a", indexed("pa", t.size())));
        v.put("WRITE", write.apply("o", "po"));
        v.put("WRITE_I", write.apply("o", indexed("po", outSize)));
        return fill("UnaryLoop", v);
    }

    /**
     * Escapes text for Javadoc, which is HTML: operation descriptions such as
     * {@code x < y} or {@code x & y} would otherwise be malformed.
     */
    private static String html(String text) {
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    private static String indexed(String base, int size) {
        return size == 1 ? base + " + i" : base + " + i * " + size;
    }

    private String fill(String template, Map<String, String> values) {
        Matcher m = PLACEHOLDER.matcher(templates.get(template));
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String value = values.get(m.group(1));
            if (value == null) {
                throw new IllegalStateException("template " + template + ": no value for ${" + m.group(1) + "}");
            }
            m.appendReplacement(sb, Matcher.quoteReplacement(value));
        }
        m.appendTail(sb);
        return sb.toString();
    }
}
