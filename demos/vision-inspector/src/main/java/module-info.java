module org.jtorch.demo.vision {
    requires org.jtorch.demo.common;
    requires transitive org.jtorch.api;
    requires transitive org.jtorch.nn;
    requires transitive org.jtorch.vision;
    requires org.jtorch.serialization;
    requires org.jtorch.json;
    requires java.desktop;

    exports org.jtorch.demo.vision;
}
