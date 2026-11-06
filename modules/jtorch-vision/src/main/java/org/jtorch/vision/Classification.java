package org.jtorch.vision;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.jtorch.core.Tensor;

/** Top-k classification results derived from a logits row. */
public record Classification(int index, String label, double confidence) {
    public static List<Classification> topK(Tensor logits, List<String> labels, int k) {
        if (logits.dim() != 1) {
            throw new IllegalArgumentException("topK expects a 1-D logits tensor");
        }
        if (k <= 0) {
            throw new IllegalArgumentException("k must be positive");
        }
        Tensor probabilities = logits.softmax(0);
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < logits.size(0); i++) {
            order.add(i);
        }
        order.sort(Comparator.comparingDouble((Integer i) -> probabilities.get(i)).reversed());
        List<Classification> results = new ArrayList<>();
        for (int i = 0; i < Math.min(k, order.size()); i++) {
            int index = order.get(i);
            String label = index < labels.size() ? labels.get(index) : "class-" + index;
            results.add(new Classification(index, label, probabilities.get(index)));
        }
        return List.copyOf(results);
    }
}
