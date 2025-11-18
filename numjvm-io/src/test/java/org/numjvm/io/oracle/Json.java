package org.numjvm.io.oracle;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal JSON reader for the oracle manifest (objects, arrays, strings without escapes
 * other than the standard ones, numbers, booleans, null). Test code only.
 */
final class Json {

    private final String s;
    private int pos;

    private Json(String s) {
        this.s = s;
    }

    static Object parse(String text) {
        Json j = new Json(text);
        Object v = j.value();
        j.space();
        if (j.pos != text.length()) {
            throw j.error("trailing characters");
        }
        return v;
    }

    private Object value() {
        space();
        char c = s.charAt(pos);
        return switch (c) {
            case '{' -> object();
            case '[' -> array();
            case '"' -> string();
            case 't' -> literal("true", Boolean.TRUE);
            case 'f' -> literal("false", Boolean.FALSE);
            case 'n' -> literal("null", null);
            default -> number();
        };
    }

    private Map<String, Object> object() {
        Map<String, Object> m = new LinkedHashMap<>();
        pos++;
        space();
        if (s.charAt(pos) == '}') {
            pos++;
            return m;
        }
        while (true) {
            space();
            String key = string();
            space();
            expect(':');
            m.put(key, value());
            space();
            if (s.charAt(pos) == ',') {
                pos++;
            } else {
                expect('}');
                return m;
            }
        }
    }

    private List<Object> array() {
        List<Object> l = new ArrayList<>();
        pos++;
        space();
        if (s.charAt(pos) == ']') {
            pos++;
            return l;
        }
        while (true) {
            l.add(value());
            space();
            if (s.charAt(pos) == ',') {
                pos++;
            } else {
                expect(']');
                return l;
            }
        }
    }

    private String string() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = s.charAt(pos++);
            if (c == '"') {
                return sb.toString();
            }
            if (c == '\\') {
                char e = s.charAt(pos++);
                switch (e) {
                    case 'n' -> sb.append('\n');
                    case 't' -> sb.append('\t');
                    case 'r' -> sb.append('\r');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'u' -> {
                        sb.append((char) Integer.parseInt(s.substring(pos, pos + 4), 16));
                        pos += 4;
                    }
                    default -> sb.append(e);
                }
            } else {
                sb.append(c);
            }
        }
    }

    private Object number() {
        int start = pos;
        while (pos < s.length() && "+-0123456789.eE".indexOf(s.charAt(pos)) >= 0) {
            pos++;
        }
        String n = s.substring(start, pos);
        if (n.isEmpty()) {
            throw error("unexpected character");
        }
        if (n.contains(".") || n.contains("e") || n.contains("E")) {
            return Double.parseDouble(n);
        }
        return Long.parseLong(n);
    }

    private Object literal(String word, Object value) {
        if (!s.startsWith(word, pos)) {
            throw error("expected " + word);
        }
        pos += word.length();
        return value;
    }

    private void expect(char c) {
        if (s.charAt(pos) != c) {
            throw error("expected '" + c + "'");
        }
        pos++;
    }

    private void space() {
        while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
            pos++;
        }
    }

    private IllegalArgumentException error(String message) {
        return new IllegalArgumentException("JSON error at " + pos + ": " + message);
    }
}
