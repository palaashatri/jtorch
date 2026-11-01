package org.jtorch.core;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import org.jtorch.core.generated.Schemas;

/** A typed handle to a canonical operator schema. Handles are created by generated {@link OpRegistry} code. */
public final class Op {
    private static final Map<String, Op> REGISTRY = new ConcurrentHashMap<>();

    private final Schemas.Schema schema;

    private Op(Schemas.Schema schema) {
        this.schema = schema;
    }

    public static Op of(String fullName) {
        return REGISTRY.computeIfAbsent(Objects.requireNonNull(fullName, "fullName"),
                name -> new Op(Schemas.get(name)));
    }

    public String name() {
        return schema.fullName();
    }

    public Schemas.Schema schema() {
        return schema;
    }

    public boolean differentiable() {
        return schema.differentiable();
    }

    /**
     * Composite operators are expanded into primitive operators by the backend. Their autograd graph is built from
     * the decomposition, exactly like PyTorch's CompositeImplicitAutograd kernels, so they do not register their own
     * backward formula.
     */
    public boolean composite() {
        return schema.composite();
    }

    public String reverse() {
        return schema.reverse();
    }

    public String namespace() {
        return schema.namespace();
    }

    @Override
    public String toString() {
        return schema.fullName();
    }
}
