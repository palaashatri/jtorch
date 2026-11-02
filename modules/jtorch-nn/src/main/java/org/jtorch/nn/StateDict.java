package org.jtorch.nn;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.jtorch.core.Tensor;

/** Ordered name-to-tensor mapping used by module state dictionaries and checkpoint formats. */
public final class StateDict extends LinkedHashMap<String, Tensor> {
    private static final long serialVersionUID = 1L;

    public StateDict() {}

    public StateDict(Map<String, Tensor> source) {
        super(source);
    }

    public Tensor require(String name) {
        Tensor tensor = get(name);
        if (tensor == null) {
            throw new IllegalArgumentException("State dict has no entry '" + name + "'; available: " + keySet());
        }
        return tensor;
    }

    public List<String> names() {
        return List.copyOf(keySet());
    }
}
