plugins {
    base
}

val javaProjects = setOf(
    "jtorch-json",
    "jtorch-generated",
    "jtorch-core",
    "jtorch-autograd",
    "jtorch-cpu-reference",
    "jtorch-api",
    "jtorch-nn",
    "jtorch-optim",
    "jtorch-data",
    "jtorch-serialization",
    "jtorch-vision",
    "jtorch-demo-common",
    "tensorlab",
    "vision-inspector",
    "parity",
    "opgen",
    "vertical-slice",
)

subprojects {
    if (name in javaProjects) {
        apply(plugin = "java-library")

        extensions.configure<JavaPluginExtension> {
            toolchain {
                languageVersion.set(JavaLanguageVersion.of(25))
            }
            withSourcesJar()
        }

        tasks.withType<JavaCompile>().configureEach {
            options.release.set(25)
            options.encoding = "UTF-8"
            options.compilerArgs.addAll(listOf("-Xlint:all", "-Werror"))
        }

        tasks.withType<Test>().configureEach {
            useJUnitPlatform()
            maxHeapSize = "1g"
            testLogging {
                events("passed", "skipped", "failed")
                showStandardStreams = true
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
        }

        dependencies {
            "testImplementation"("org.junit.jupiter:junit-jupiter:5.11.4")
            "testRuntimeOnly"("org.junit.platform:junit-platform-launcher")
        }
    }
}

tasks.named("check") {
    dependsOn(subprojects.filter { it.name in javaProjects }.map { "${it.path}:check" })
}

tasks.register("demoCheck") {
    group = "verification"
    description = "Runs headless smoke tests for every demo application."
    dependsOn(":demos:tensorlab:demoCheck", ":demos:vision-inspector:demoCheck")
}

tasks.register("demoPackage") {
    group = "distribution"
    description = "Builds runtime images for every demo application."
    dependsOn(":demos:tensorlab:demoPackage", ":demos:vision-inspector:demoPackage")
}

tasks.register("regenerateSchemas") {
    group = "build"
    description = "Regenerates operator metadata, docs and parity descriptors from schema/."
    dependsOn(":tools:opgen:run")
}

tasks.register("parityFast") {
    group = "verification"
    description = "Runs the pinned PyTorch differential parity suite."
    dependsOn(":tests:parity:test")
}

tasks.register("verifyGeneratedClean") {
    group = "verification"
    description = "Fails when generated sources differ from schema inputs."
    dependsOn(":tools:opgen:verifyGeneratedClean")
}
