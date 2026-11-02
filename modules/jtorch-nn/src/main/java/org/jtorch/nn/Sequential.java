package org.jtorch.nn;

import java.util.ArrayList;
import java.util.List;
import org.jtorch.core.Tensor;

/** Runs modules in registration order, matching {@code torch.nn.Sequential}. */
public final class Sequential extends Module {
    public Sequential(Module... modules) {
        for (int i = 0; i < modules.length; i++) {
            registerModule(Integer.toString(i), modules[i]);
        }
    }

    public Sequential(List<Module> modules) {
        this(modules.toArray(Module[]::new));
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor value = input;
        for (Module module : modules()) {
            value = module.call(value);
        }
        return value;
    }

    public List<Module> modules() {
        List<Module> result = new ArrayList<>();
        for (int i = 0; i < Integer.MAX_VALUE; i++) {
            Module child = child(Integer.toString(i));
            if (child == null) {
                break;
            }
            result.add(child);
        }
        return List.copyOf(result);
    }

    private Module child(String name) {
        return namedChildren().get(name);
    }
}
