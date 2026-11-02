module org.jtorch.autograd {
    requires transitive org.jtorch.core;

    exports org.jtorch.autograd;

    provides org.jtorch.core.spi.AutogradProvider with org.jtorch.autograd.AutogradProviderImpl;
}
