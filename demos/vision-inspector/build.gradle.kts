plugins {
    application
}

dependencies {
    implementation(project(":demos:jtorch-demo-common"))
    implementation(project(":modules:jtorch-api"))
    implementation(project(":modules:jtorch-nn"))
    implementation(project(":modules:jtorch-vision"))
    implementation(project(":modules:jtorch-serialization"))
    implementation(project(":modules:jtorch-json"))
}

application {
    mainModule.set("org.jtorch.demo.vision")
    mainClass.set("org.jtorch.demo.vision.VisionInspectorApp")
}

java {
    modularity.inferModulePath.set(true)
}

val reportFile = layout.buildDirectory.file("demo/vision-inspector-report.json")

val demoCheck = tasks.register<JavaExec>("demoCheck") {
    group = "verification"
    description = "Runs Vision Inspector's deterministic headless fixture smoke test."
    dependsOn(tasks.named("classes"))
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.vision.VisionInspectorApp")
    args("--smoke-test", "--headless", "--output", reportFile.get().asFile.absolutePath)
}

val packageInput = layout.buildDirectory.dir("package/input")

val copyRuntime = tasks.register<Sync>("copyRuntime") {
    dependsOn(tasks.named("jar"))
    from(tasks.named("jar"))
    from(configurations.runtimeClasspath)
    into(packageInput)
}

tasks.register<JavaExec>("demoPackage") {
    group = "distribution"
    description = "Builds a jpackage app image and runs its packaged headless smoke test."
    dependsOn(copyRuntime)
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.demo.vision.PackageTool")
    args(
        packageInput.get().asFile.absolutePath,
        layout.buildDirectory.dir("package").get().asFile.absolutePath,
        reportFile.get().asFile.absolutePath)
}
