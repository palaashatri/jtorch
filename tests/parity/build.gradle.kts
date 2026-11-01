plugins {
    `java-library`
}

dependencies {
    testImplementation(project(":modules:jtorch-api"))
    testImplementation(project(":modules:jtorch-json"))
    testImplementation(project(":modules:jtorch-core"))
    testImplementation(project(":modules:jtorch-autograd"))
    testImplementation(project(":backends:jtorch-cpu-reference"))
}

sourceSets {
    test {
        java.srcDir("src/test/generated/java")
    }
}

tasks.named<Test>("test") {
    // Golden fixtures are capped at a few kilobytes; keep the default heap.
}
