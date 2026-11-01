package org.jtorch.autograd;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.jtorch.core.AutogradException;
import org.jtorch.core.GradMode;
import org.jtorch.core.Tensor;
import org.jtorch.core.autograd.Node;
import org.jtorch.core.spi.GradientEngine;

/** Reverse-mode autograd engine: topological traversal, leaf accumulation, version checks and graph release. */
public final class AutogradEngine implements GradientEngine {
    public AutogradEngine() {}

    @Override
    public void backward(Tensor root, Tensor gradient, boolean retainGraph, boolean createGraph) {
        if (createGraph) {
            throw new AutogradException("create_graph (higher-order gradients) is not implemented yet");
        }
        if (!root.requiresGrad()) {
            throw new AutogradException("element 0 of tensors does not require grad and does not have a grad_fn");
        }
        Node rootNode = root.impl().gradFn();
        if (rootNode == null && root.impl().isGraphFreed()) {
            throw new AutogradException(
                    "Trying to backward through the graph a second time; pass retainGraph=true to keep it");
        }
        Tensor seed = seed(root, gradient);
        GradMode.noGrad(() -> run(root, rootNode, seed, retainGraph));
    }

    private static Tensor seed(Tensor root, Tensor gradient) {
        if (gradient == null) {
            if (root.numel() != 1) {
                throw new AutogradException(
                        "grad can be implicitly created only for scalar outputs, but the output has "
                                + root.numel() + " elements");
            }
            return Tensor.onesLike(root);
        }
        if (!gradient.shape().equals(root.shape())) {
            throw new AutogradException(
                    "Mismatch in shape of grad_output: expected " + root.shape() + " but got " + gradient.shape());
        }
        return gradient;
    }

    private static void run(Tensor root, Node rootNode, Tensor seed, boolean retainGraph) {
        if (rootNode == null) {
            accumulateLeaf(root, seed);
            return;
        }
        List<Node> order = topological(rootNode);
        Map<Node, Tensor[]> gradients = new IdentityHashMap<>();
        Tensor[] rootGradients = new Tensor[rootNode.outputCount()];
        rootGradients[root.impl().outputNr()] = seed;
        gradients.put(rootNode, rootGradients);

        for (Node node : order) {
            Tensor[] accumulated = gradients.get(node);
            if (accumulated == null) {
                continue;
            }
            node.validateSavedTensors();
            Tensor[] outputs = node.outputs();
            for (int i = 0; i < accumulated.length; i++) {
                if (accumulated[i] != null && i < outputs.length) {
                    for (var hook : outputs[i].impl().postAccumulateHooks()) {
                        hook.accept(accumulated[i]);
                    }
                }
            }
            Tensor[] filled = new Tensor[node.outputCount()];
            for (int i = 0; i < filled.length; i++) {
                filled[i] = accumulated[i] != null ? accumulated[i] : Tensor.zerosLike(outputs[i]);
            }
            Tensor[] inputGradients = node.backward(filled);
            Tensor[] inputs = node.inputs();
            for (int i = 0; i < inputs.length; i++) {
                Tensor input = inputs[i];
                Tensor inputGradient = i < inputGradients.length ? inputGradients[i] : null;
                if (inputGradient == null || !input.requiresGrad()) {
                    continue;
                }
                if (inputGradient.dtype() != input.dtype()) {
                    // PyTorch accumulates each input gradient in that input's dtype even when a formula computes in
                    // a promoted dtype.
                    inputGradient = inputGradient.to(input.dtype());
                }
                Node producer = input.impl().gradFn();
                if (producer == null) {
                    accumulateLeaf(input, inputGradient);
                } else {
                    Tensor[] producerGradients = gradients.computeIfAbsent(
                            producer, ignored -> new Tensor[producer.outputCount()]);
                    int outputNr = input.impl().outputNr();
                    producerGradients[outputNr] = producerGradients[outputNr] == null
                            ? inputGradient
                            : producerGradients[outputNr].add(inputGradient);
                }
            }
            if (!retainGraph) {
                for (Tensor output : outputs) {
                    if (output.impl().gradFn() == node) {
                        output.impl().clearGradFn();
                    }
                }
            }
        }
    }

    private static void accumulateLeaf(Tensor leaf, Tensor gradient) {
        Tensor existing = leaf.grad();
        // PyTorch's AccumulateGrad materialises an independent, contiguous .grad on first accumulation; incoming
        // gradients are frequently broadcast views (stride 0) or slices of a shared grad_output.
        Tensor accumulated = existing == null ? gradient.detach().clone() : existing.add(gradient);
        leaf.impl().setGrad(accumulated);
        for (var hook : leaf.impl().postAccumulateHooks()) {
            hook.accept(accumulated);
        }
    }

    private static List<Node> topological(Node root) {
        Set<Node> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Node> order = new ArrayList<>();
        visit(root, visited, order);
        Collections.reverse(order);
        return order;
    }

    private static void visit(Node node, Set<Node> visited, List<Node> order) {
        if (!visited.add(node)) {
            return;
        }
        for (Tensor input : node.inputs()) {
            Node producer = input.impl().gradFn();
            if (producer != null) {
                visit(producer, visited, order);
            }
        }
        order.add(node);
    }

    /** Diagnostic: node names in reverse topological order, used by tests and the profiler. */
    public static List<String> graphOrder(Tensor root) {
        Node rootNode = root.impl().gradFn();
        if (rootNode == null) {
            return List.of();
        }
        Set<String> names = new LinkedHashSet<>();
        for (Node node : topological(rootNode)) {
            names.add(node.name());
        }
        return List.copyOf(names);
    }
}
