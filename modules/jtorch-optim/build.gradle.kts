plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-nn"))
    testImplementation(project(":modules:jtorch-json"))
}

java {
    modularity.inferModulePath.set(true)
}
