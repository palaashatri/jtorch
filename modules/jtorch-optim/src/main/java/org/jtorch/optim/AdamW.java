package org.jtorch.optim;

/** Adam with decoupled weight decay, matching {@code torch.optim.AdamW}. */
public final class AdamW extends Adam {
    private AdamW(Config config) {
        super(config);
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Builder that marks the configuration as decoupled. */
    public static final class Builder extends Adam.Builder {
        public Builder() {}

        @Override
        public Builder parameters(java.util.List<org.jtorch.nn.Parameter> value) {
            super.parameters(value);
            return this;
        }

        @Override
        public Builder learningRate(double value) {
            super.learningRate(value);
            return this;
        }

        @Override
        public Builder betas(double first, double second) {
            super.betas(first, second);
            return this;
        }

        @Override
        public Builder epsilon(double value) {
            super.epsilon(value);
            return this;
        }

        @Override
        public Builder weightDecay(double value) {
            super.weightDecay(value);
            return this;
        }

        @Override
        protected Adam create(Config config) {
            return new AdamW(new Config(
                    config.parameters(),
                    config.learningRate(),
                    config.beta1(),
                    config.beta2(),
                    config.epsilon(),
                    config.weightDecay(),
                    true));
        }

        @Override
        public AdamW build() {
            return (AdamW) super.build();
        }
    }
}
