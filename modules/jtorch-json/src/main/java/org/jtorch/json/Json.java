package org.jtorch.json;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal, dependency-free JSON reader and writer used by JTorch checkpoints, Safetensors headers and
 * compatibility fixtures.
 *
 * <p>Parsed values are {@link Map}, {@link List}, {@link String}, {@link Long}, {@link Double},
 * {@link Boolean} or {@code null}. Numbers without a fraction or exponent become {@link Long} when they fit,
 * otherwise {@link Double}. The writer emits shortest round-trip doubles so that golden fixtures preserve
 * exact values.
 */
public final class Json {
    private Json() {}

    public static Object parse(String text) {
        Parser parser = new Parser(text);
        Object value = parser.value();
        parser.skipWhitespace();
        if (!parser.atEnd()) {
            throw new JsonException("Trailing content at offset " + parser.offset());
        }
        return value;
    }

    public static Map<String, Object> parseObject(String text) {
        Object value = parse(text);
        if (value instanceof Map<?, ?> map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> typed = (Map<String, Object>) map;
            return typed;
        }
        throw new JsonException("Expected a JSON object, got " + typeName(value));
    }

    public static String write(Object value) {
        StringBuilder out = new StringBuilder();
        writeValue(value, out, -1, 0);
        return out.toString();
    }

    public static String writePretty(Object value) {
        StringBuilder out = new StringBuilder();
        writeValue(value, out, 2, 0);
        return out.toString();
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(Object value) {
        if (value instanceof Map<?, ?> map) {
            return (Map<String, Object>) map;
        }
        throw new JsonException("Expected object, got " + typeName(value));
    }

    @SuppressWarnings("unchecked")
    public static List<Object> array(Object value) {
        if (value instanceof List<?> list) {
            return (List<Object>) list;
        }
        throw new JsonException("Expected array, got " + typeName(value));
    }

    public static String string(Object value) {
        if (value instanceof String string) {
            return string;
        }
        throw new JsonException("Expected string, got " + typeName(value));
    }

    public static long longValue(Object value) {
        if (value instanceof Long number) {
            return number;
        }
        if (value instanceof Double number && number == Math.rint(number)) {
            return number.longValue();
        }
        throw new JsonException("Expected integer, got " + typeName(value));
    }

    public static int intValue(Object value) {
        long number = longValue(value);
        if (number < Integer.MIN_VALUE || number > Integer.MAX_VALUE) {
            throw new JsonException("Integer out of range: " + number);
        }
        return (int) number;
    }

    public static double doubleValue(Object value) {
        if (value instanceof Double number) {
            return number;
        }
        if (value instanceof Long number) {
            return number.doubleValue();
        }
        throw new JsonException("Expected number, got " + typeName(value));
    }

    public static boolean booleanValue(Object value) {
        if (value instanceof Boolean bool) {
            return bool;
        }
        throw new JsonException("Expected boolean, got " + typeName(value));
    }

    public static Object get(Map<String, Object> object, String key) {
        return object.get(key);
    }

    public static Object require(Map<String, Object> object, String key) {
        if (!object.containsKey(key)) {
            throw new JsonException("Missing key '" + key + "'");
        }
        return object.get(key);
    }

    private static void writeValue(Object value, StringBuilder out, int indent, int depth) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String string) {
            writeString(string, out);
        } else if (value instanceof Boolean bool) {
            out.append(bool);
        } else if (value instanceof Double number) {
            writeDouble(number, out);
        } else if (value instanceof Float number) {
            writeDouble(number.doubleValue(), out);
        } else if (value instanceof Long || value instanceof Integer || value instanceof Short
                || value instanceof Byte) {
            out.append(((Number) value).longValue());
        } else if (value instanceof Number number) {
            writeDouble(number.doubleValue(), out);
        } else if (value instanceof Map<?, ?> map) {
            writeObject(map, out, indent, depth);
        } else if (value instanceof List<?> list) {
            writeArray(list, out, indent, depth);
        } else {
            throw new JsonException("Unsupported JSON value type: " + value.getClass().getName());
        }
    }

    private static void writeDouble(double value, StringBuilder out) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new JsonException("JSON cannot represent non-finite number: " + value);
        }
        out.append(Double.toString(value));
    }

    private static void writeObject(Map<?, ?> map, StringBuilder out, int indent, int depth) {
        if (map.isEmpty()) {
            out.append("{}");
            return;
        }
        out.append('{');
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new JsonException("Object keys must be strings");
            }
            if (!first) {
                out.append(',');
            }
            first = false;
            newline(out, indent, depth + 1);
            writeString(key, out);
            out.append(':');
            if (indent >= 0) {
                out.append(' ');
            }
            writeValue(entry.getValue(), out, indent, depth + 1);
        }
        newline(out, indent, depth);
        out.append('}');
    }

    private static void writeArray(List<?> list, StringBuilder out, int indent, int depth) {
        if (list.isEmpty()) {
            out.append("[]");
            return;
        }
        out.append('[');
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) {
                out.append(',');
            }
            newline(out, indent, depth + 1);
            writeValue(list.get(i), out, indent, depth + 1);
        }
        newline(out, indent, depth);
        out.append(']');
    }

    private static void newline(StringBuilder out, int indent, int depth) {
        if (indent < 0) {
            return;
        }
        out.append('\n');
        out.append(" ".repeat(indent * depth));
    }

    private static void writeString(String value, StringBuilder out) {
        out.append('"');
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            switch (ch) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (ch < 0x20) {
                        out.append(String.format("\\u%04x", (int) ch));
                    } else {
                        out.append(ch);
                    }
                }
            }
        }
        out.append('"');
    }

    private static String typeName(Object value) {
        return value == null ? "null" : value.getClass().getSimpleName();
    }

    private static final class Parser {
        private final String text;
        private int index;

        Parser(String text) {
            this.text = text;
        }

        int offset() {
            return index;
        }

        boolean atEnd() {
            return index >= text.length();
        }

        void skipWhitespace() {
            while (index < text.length()) {
                char ch = text.charAt(index);
                if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') {
                    index++;
                } else {
                    break;
                }
            }
        }

        Object value() {
            skipWhitespace();
            if (atEnd()) {
                throw new JsonException("Unexpected end of input");
            }
            char ch = text.charAt(index);
            return switch (ch) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                case 't' -> literal("true", Boolean.TRUE);
                case 'f' -> literal("false", Boolean.FALSE);
                case 'n' -> literal("null", null);
                default -> number();
            };
        }

        private Object object() {
            expect('{');
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (peek() == '}') {
                index++;
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = string();
                skipWhitespace();
                expect(':');
                Object value = value();
                if (map.put(key, value) != null) {
                    throw new JsonException("Duplicate object key '" + key + "'");
                }
                skipWhitespace();
                char ch = next();
                if (ch == '}') {
                    return map;
                }
                if (ch != ',') {
                    throw new JsonException("Expected ',' or '}' at offset " + (index - 1));
                }
            }
        }

        private List<Object> array() {
            expect('[');
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (peek() == ']') {
                index++;
                return list;
            }
            while (true) {
                list.add(value());
                skipWhitespace();
                char ch = next();
                if (ch == ']') {
                    return list;
                }
                if (ch != ',') {
                    throw new JsonException("Expected ',' or ']' at offset " + (index - 1));
                }
            }
        }

        private String string() {
            expect('"');
            StringBuilder out = new StringBuilder();
            while (true) {
                if (atEnd()) {
                    throw new JsonException("Unterminated string");
                }
                char ch = text.charAt(index++);
                if (ch == '"') {
                    return out.toString();
                }
                if (ch != '\\') {
                    out.append(ch);
                    continue;
                }
                if (atEnd()) {
                    throw new JsonException("Unterminated escape sequence");
                }
                char escaped = text.charAt(index++);
                switch (escaped) {
                    case '"' -> out.append('"');
                    case '\\' -> out.append('\\');
                    case '/' -> out.append('/');
                    case 'b' -> out.append('\b');
                    case 'f' -> out.append('\f');
                    case 'n' -> out.append('\n');
                    case 'r' -> out.append('\r');
                    case 't' -> out.append('\t');
                    case 'u' -> {
                        if (index + 4 > text.length()) {
                            throw new JsonException("Truncated unicode escape");
                        }
                        out.append((char) Integer.parseInt(text.substring(index, index + 4), 16));
                        index += 4;
                    }
                    default -> throw new JsonException("Invalid escape '\\" + escaped + "'");
                }
            }
        }

        private Object number() {
            int start = index;
            if (peek() == '-') {
                index++;
            }
            while (!atEnd() && Character.isDigit(text.charAt(index))) {
                index++;
            }
            boolean floating = false;
            if (!atEnd() && text.charAt(index) == '.') {
                floating = true;
                index++;
                while (!atEnd() && Character.isDigit(text.charAt(index))) {
                    index++;
                }
            }
            if (!atEnd() && (text.charAt(index) == 'e' || text.charAt(index) == 'E')) {
                floating = true;
                index++;
                if (!atEnd() && (text.charAt(index) == '+' || text.charAt(index) == '-')) {
                    index++;
                }
                while (!atEnd() && Character.isDigit(text.charAt(index))) {
                    index++;
                }
            }
            if (index == start || (index == start + 1 && text.charAt(start) == '-')) {
                throw new JsonException("Expected a value at offset " + start);
            }
            String token = text.substring(start, index);
            try {
                if (!floating) {
                    return Long.parseLong(token);
                }
                return Double.parseDouble(token);
            } catch (NumberFormatException error) {
                throw new JsonException("Invalid number '" + token + "'");
            }
        }

        private Object literal(String token, Object value) {
            if (!text.startsWith(token, index)) {
                throw new JsonException("Invalid literal at offset " + index);
            }
            index += token.length();
            return value;
        }

        private char peek() {
            return atEnd() ? '\0' : text.charAt(index);
        }

        private char next() {
            if (atEnd()) {
                throw new JsonException("Unexpected end of input");
            }
            return text.charAt(index++);
        }

        private void expect(char expected) {
            char actual = next();
            if (actual != expected) {
                throw new JsonException(
                        "Expected '" + expected + "' but found '" + actual + "' at offset " + (index - 1));
            }
        }
    }
}
