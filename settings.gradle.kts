rootProject.name = "jtorch"

dependencyResolutionManagement {
    repositories {
        mavenCentral()
    }
}

include(
    "modules:jtorch-json",
    "modules:jtorch-generated",
    "modules:jtorch-core",
    "modules:jtorch-autograd",
    "backends:jtorch-cpu-reference",
    "modules:jtorch-api",
    "modules:jtorch-nn",
    "modules:jtorch-optim",
    "modules:jtorch-data",
    "modules:jtorch-serialization",
    "demos:jtorch-demo-common",
    "demos:tensorlab",
    "tests:parity",
    "tools:opgen",
    "examples:vertical-slice",
)
