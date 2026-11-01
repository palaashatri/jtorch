package org.jtorch.nn;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import org.jtorch.api.Torch;
import org.jtorch.core.Tensor;

/**
 * Base class for neural-network modules. Parameters, buffers and child modules are registered explicitly; discovery
 * never relies on reflection.
 */
public abstract class Module {
    protected Module() {}

    private final LinkedHashMap<String, Module> children = new LinkedHashMap<>();
    private final LinkedHashMap<String, Parameter> parameters = new LinkedHashMap<>();
    private final LinkedHashMap<String, Tensor> persistentBuffers = new LinkedHashMap<>();
    private final LinkedHashMap<String, Tensor> nonPersistentBuffers = new LinkedHashMap<>();
    private final List<UnaryOperator<Tensor>> forwardPreHooks = new ArrayList<>();
    private final List<UnaryOperator<Tensor>> forwardHooks = new ArrayList<>();
    private boolean training = true;

    protected <M extends Module> M registerModule(String name, M module) {
        if (children.putIfAbsent(name, module) != null) {
            throw new IllegalArgumentException("Module '" + name + "' is already registered");
        }
        return module;
    }

    protected Parameter registerParameter(String name, Parameter parameter) {
        if (parameters.putIfAbsent(name, parameter) != null) {
            throw new IllegalArgumentException("Parameter '" + name + "' is already registered");
        }
        return parameter;
    }

    protected Tensor registerBuffer(String name, Tensor tensor, boolean persistent) {
        if (persistentBuffers.containsKey(name) || nonPersistentBuffers.containsKey(name)
                || parameters.containsKey(name) || children.containsKey(name)) {
            throw new IllegalArgumentException("Buffer '" + name + "' is already registered");
        }
        if (persistent) {
            persistentBuffers.put(name, tensor);
        } else {
            nonPersistentBuffers.put(name, tensor);
        }
        return tensor;
    }

    /** Subclasses implement the numerical forward pass. */
    public abstract Tensor forward(Tensor input);

    /** Hook-aware invocation; prefer this over {@link #forward(Tensor)} at application boundaries. */
    public final Tensor call(Tensor input) {
        Tensor value = input;
        for (UnaryOperator<Tensor> hook : forwardPreHooks) {
            value = hook.apply(value);
        }
        value = forward(value);
        for (UnaryOperator<Tensor> hook : forwardHooks) {
            value = hook.apply(value);
        }
        return value;
    }

    public final Module registerForwardPreHook(UnaryOperator<Tensor> hook) {
        forwardPreHooks.add(hook);
        return this;
    }

    public final Module registerForwardHook(UnaryOperator<Tensor> hook) {
        forwardHooks.add(hook);
        return this;
    }

    public final Module train(boolean value) {
        training = value;
        for (Module child : children.values()) {
            child.train(value);
        }
        return this;
    }

    public final Module train() {
        return train(true);
    }

    public final Module eval() {
        return train(false);
    }

    public final boolean isTraining() {
        return training;
    }

    protected final Map<String, Module> namedChildren() {
        return java.util.Collections.unmodifiableMap(children);
    }

    public final List<Parameter> parameters() {
        return List.copyOf(namedParameters().values());
    }

    public final Map<String, Parameter> namedParameters() {
        LinkedHashMap<String, Parameter> result = new LinkedHashMap<>();
        java.util.Set<org.jtorch.core.TensorImpl> seen =
                java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        collectParameters("", result, seen);
        return result;
    }

    public final List<Tensor> buffers() {
        return List.copyOf(namedBuffers().values());
    }

    public final Map<String, Tensor> namedBuffers() {
        LinkedHashMap<String, Tensor> result = new LinkedHashMap<>();
        collectBuffers("", result);
        return result;
    }

    public final StateDict stateDict() {
        StateDict state = new StateDict();
        collectState("", state);
        return state;
    }

    public final LoadStateResult loadStateDict(StateDict state, boolean strict) {
        Map<String, Tensor> expected = new LinkedHashMap<>();
        collectStateInto("", expected);
        List<String> missing = new ArrayList<>();
        List<String> unexpected = new ArrayList<>();
        for (String name : expected.keySet()) {
            if (!state.containsKey(name)) {
                missing.add(name);
            }
        }
        for (String name : state.keySet()) {
            if (!expected.containsKey(name)) {
                unexpected.add(name);
            }
        }
        if (strict && (!missing.isEmpty() || !unexpected.isEmpty())) {
            throw new IllegalArgumentException("Error(s) in loading state_dict: missing keys " + missing
                    + ", unexpected keys " + unexpected);
        }
        Torch.noGrad(() -> {
            for (Map.Entry<String, Tensor> entry : expected.entrySet()) {
                Tensor source = state.get(entry.getKey());
                if (source == null) {
                    continue;
                }
                Tensor target = entry.getValue();
                if (!source.shape().equals(target.shape())) {
                    throw new IllegalArgumentException("Shape mismatch for '" + entry.getKey() + "': expected "
                            + target.shape() + " but got " + source.shape());
                }
                target.copy_(source.to(target.dtype()));
            }
        });
        return new LoadStateResult(missing, unexpected);
    }

    public final void zeroGrad() {
        for (Parameter parameter : parameters()) {
            parameter.impl().setGrad(null);
        }
    }

    private void collectParameters(
            String prefix, Map<String, Parameter> out, java.util.Set<org.jtorch.core.TensorImpl> seen) {
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            if (seen.add(entry.getValue().impl())) {
                out.put(prefix + entry.getKey(), entry.getValue());
            }
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            entry.getValue().collectParameters(prefix + entry.getKey() + ".", out, seen);
        }
    }

    private void collectStateInto(String prefix, Map<String, Tensor> out) {
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Tensor> entry : persistentBuffers.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            entry.getValue().collectStateInto(prefix + entry.getKey() + ".", out);
        }
    }

    private void collectBuffers(String prefix, Map<String, Tensor> out) {
        for (Map.Entry<String, Tensor> entry : persistentBuffers.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Tensor> entry : nonPersistentBuffers.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            entry.getValue().collectBuffers(prefix + entry.getKey() + ".", out);
        }
    }

    private void collectState(String prefix, StateDict out) {
        for (Map.Entry<String, Parameter> entry : parameters.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Tensor> entry : persistentBuffers.entrySet()) {
            out.put(prefix + entry.getKey(), entry.getValue());
        }
        for (Map.Entry<String, Module> entry : children.entrySet()) {
            entry.getValue().collectState(prefix + entry.getKey() + ".", out);
        }
    }
}
