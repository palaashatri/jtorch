plugins {
    application
}

dependencies {
    implementation(project(":modules:jtorch-json"))
}

application {
    mainClass.set("org.jtorch.opgen.Main")
}

tasks.named<JavaExec>("run") {
    workingDir = rootProject.projectDir
    args("--root", rootProject.projectDir.absolutePath)
}

val verifyGeneratedClean = tasks.register<JavaExec>("verifyGeneratedClean") {
    group = "verification"
    description = "Re-renders schema outputs in memory and fails when checked-in files differ."
    classpath = sourceSets["main"].runtimeClasspath
    mainClass.set("org.jtorch.opgen.Main")
    workingDir = rootProject.projectDir
    args("--check", "--root", rootProject.projectDir.absolutePath)
}
