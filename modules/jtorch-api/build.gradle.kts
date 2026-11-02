plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-core"))
    api(project(":modules:jtorch-autograd"))
    api(project(":backends:jtorch-cpu-reference"))

    testImplementation(project(":modules:jtorch-json"))
}

java {
    modularity.inferModulePath.set(true)
}
