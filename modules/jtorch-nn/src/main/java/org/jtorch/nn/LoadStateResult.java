package org.jtorch.nn;

import java.util.List;

/** Result of a state-dictionary load, mirroring PyTorch's missing/unexpected key report. */
public record LoadStateResult(List<String> missingKeys, List<String> unexpectedKeys) {
    public LoadStateResult {
        missingKeys = List.copyOf(missingKeys);
        unexpectedKeys = List.copyOf(unexpectedKeys);
    }

    public boolean isClean() {
        return missingKeys.isEmpty() && unexpectedKeys.isEmpty();
    }
}
