plugins {
    `java-library`
}

dependencies {
    api(project(":modules:jtorch-api"))
    api(project(":modules:jtorch-nn"))
    implementation(project(":modules:jtorch-serialization"))
    testImplementation(project(":modules:jtorch-serialization"))
}

java {
    modularity.inferModulePath.set(true)
}
