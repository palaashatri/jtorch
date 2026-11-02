module org.jtorch.demo.tensorlab {
    requires org.jtorch.demo.common;
    requires transitive org.jtorch.api;
    requires transitive org.jtorch.nn;
    requires transitive org.jtorch.optim;
    requires transitive org.jtorch.data;
    requires transitive org.jtorch.serialization;
    requires java.desktop;

    exports org.jtorch.demo.tensorlab;
}
